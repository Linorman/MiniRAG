"""
Qdrant Storage Module
=====================

This module provides a storage interface for vector databases using Qdrant,
a high-performance vector search engine designed for production-ready vector
similarity search with extended filtering support.

Author: MiniRAG team
License: MIT

Features:
    - Async/sync support for all operations
    - Efficient batch processing for embeddings
    - Configurable distance metrics (Cosine, Euclidean, Dot)
    - Payload filtering support
    - Local and remote (cloud) deployment support
    - gRPC communication support for faster performance

Dependencies:
    - qdrant-client
    - numpy
    - asyncio

Usage:
    from minirag.kg.qdrant_impl import QdrantVectorDBStorage
"""

import asyncio
import os
import uuid
from dataclasses import dataclass

import numpy as np
import pipmaster as pm
from tqdm.asyncio import tqdm as tqdm_async

from minirag.base import BaseVectorStorage
from minirag.utils import logger

if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
)


@dataclass
class QdrantVectorDBStorage(BaseVectorStorage):
    """Qdrant vector storage implementation."""

    cosine_better_than_threshold: float = float(os.getenv("COSINE_THRESHOLD", "0.2"))

    def __post_init__(self):
        try:
            # Use global config value if specified, otherwise use default
            config = self.global_config.get("vector_db_storage_cls_kwargs", {})
            self.cosine_better_than_threshold = config.get(
                "cosine_better_than_threshold", self.cosine_better_than_threshold
            )

            # Configuration options
            use_async = config.get("use_async", False)
            use_memory = config.get("use_memory", False)
            use_grpc = config.get("prefer_grpc", False)

            # Connection parameters
            host = config.get("host", "localhost")
            port = config.get("port", 6333)
            grpc_port = config.get("grpc_port", 6334)
            url = config.get("url", None)
            api_key = config.get("api_key", None)
            
            # Local storage path
            local_path = config.get(
                "path",
                os.path.join(self.global_config["working_dir"], "qdrant_storage")
            )

            # Distance metric configuration
            distance_metric_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT,
                "manhattan": Distance.MANHATTAN,
            }
            distance_metric = config.get("distance_metric", "cosine").lower()
            self._distance = distance_metric_map.get(distance_metric, Distance.COSINE)

            # Collection settings
            self._collection_name = self.namespace
            self._vector_size = self.embedding_func.embedding_dim
            self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

            # HNSW index parameters
            self._hnsw_config = {
                "m": config.get("hnsw_m", 16),
                "ef_construct": config.get("hnsw_ef_construct", 100),
            }

            # Initialize client based on configuration
            if use_async:
                # Async client initialization
                if use_memory:
                    self._client = AsyncQdrantClient(":memory:")
                elif url:
                    self._client = AsyncQdrantClient(
                        url=url,
                        api_key=api_key,
                        prefer_grpc=use_grpc,
                    )
                else:
                    self._client = AsyncQdrantClient(
                        path=local_path,
                    )
                self._is_async = True
            else:
                # Sync client initialization
                if use_memory:
                    self._client = QdrantClient(":memory:")
                elif url:
                    self._client = QdrantClient(
                        url=url,
                        api_key=api_key,
                        prefer_grpc=use_grpc,
                    )
                elif use_grpc:
                    self._client = QdrantClient(
                        host=host,
                        grpc_port=grpc_port,
                        prefer_grpc=True,
                    )
                else:
                    if host == "localhost" and not url:
                        # Use local file-based storage
                        self._client = QdrantClient(path=local_path)
                    else:
                        self._client = QdrantClient(
                            host=host,
                            port=port,
                        )
                self._is_async = False

            # Create collection if it doesn't exist
            self._initialize_collection()

            logger.info(
                f"Qdrant client initialized for collection '{self._collection_name}' "
                f"with vector size {self._vector_size} and distance metric {distance_metric}"
            )

        except Exception as e:
            logger.error(f"Qdrant initialization failed: {str(e)}")
            raise

    def _initialize_collection(self):
        """Initialize collection with proper configuration."""
        try:
            if self._is_async:
                # For async client, we need to handle this differently
                # Create a temporary sync client for initialization
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._async_initialize_collection())
            else:
                if not self._client.collection_exists(self._collection_name):
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=self._vector_size,
                            distance=self._distance,
                            hnsw_config=self._hnsw_config,
                        ),
                    )
                    logger.info(f"Created collection '{self._collection_name}'")
                else:
                    logger.info(f"Collection '{self._collection_name}' already exists")
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    async def _async_initialize_collection(self):
        """Async version of collection initialization."""
        if not await self._client.collection_exists(self._collection_name):
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                    hnsw_config=self._hnsw_config,
                ),
            )
            logger.info(f"Created collection '{self._collection_name}'")
        else:
            logger.info(f"Collection '{self._collection_name}' already exists")

    async def upsert(self, data: dict[str, dict]):
        """Insert or update vectors in the collection."""
        if not data:
            logger.warning("Empty data provided to vector DB")
            return []

        try:
            logger.info(f"Inserting {len(data)} vectors to {self._collection_name}")

            # Prepare data
            ids = list(data.keys())
            contents = [v["content"] for v in data.values()]
            
            # Extract metadata (payloads)
            payloads = [
                {k: v for k, v in item.items() if k in self.meta_fields and k != "content"}
                or {}
                for item in data.values()
            ]

            # Generate embeddings in batches
            batches = [
                contents[i : i + self._max_batch_size]
                for i in range(0, len(contents), self._max_batch_size)
            ]

            async def wrapped_task(batch):
                result = await self.embedding_func(batch)
                pbar.update(1)
                return result

            embedding_tasks = [wrapped_task(batch) for batch in batches]
            pbar = tqdm_async(
                total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
            )
            embeddings_list = await asyncio.gather(*embedding_tasks)
            pbar.close()

            embeddings = np.concatenate(embeddings_list)

            # Prepare points for Qdrant
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i])),  # Ensure valid UUID
                    vector=embeddings[i].tolist(),
                    payload={**payloads[i], "original_id": ids[i], "content": contents[i]},
                )
                for i in range(len(ids))
            ]

            # Upsert in batches to avoid overwhelming the server
            batch_size = self._max_batch_size
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                
                if self._is_async:
                    await self._client.upsert(
                        collection_name=self._collection_name,
                        points=batch_points,
                        wait=True,
                    )
                else:
                    self._client.upsert(
                        collection_name=self._collection_name,
                        points=batch_points,
                        wait=True,
                    )

            logger.info(f"Successfully upserted {len(points)} points")
            return ids

        except Exception as e:
            logger.error(f"Error during Qdrant upsert: {str(e)}")
            raise

    async def query(self, query: str, top_k=5) -> list[dict]:
        """Query similar vectors from the collection."""
        try:
            # Generate query embedding
            embedding = await self.embedding_func([query])
            query_vector = embedding[0].tolist()

            logger.info(
                f"Querying Qdrant: top_k={top_k}, "
                f"threshold={self.cosine_better_than_threshold}"
            )

            # Perform search
            if self._is_async:
                search_result = await self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=top_k * 2,  # Request more to filter by threshold
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                search_result = self._client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=top_k * 2,
                    with_payload=True,
                    with_vectors=False,
                )

            # Process and filter results
            # Qdrant returns similarity score (higher is better for cosine)
            # We need to convert based on distance metric
            results = []
            for point in search_result:
                # For cosine distance, Qdrant returns similarity (1 = identical)
                # Convert to distance representation if needed
                if self._distance == Distance.COSINE:
                    # Qdrant returns cosine similarity, keep as is
                    score = point.score
                else:
                    # For other metrics, score is already a distance
                    score = point.score

                # Filter by threshold (for cosine, higher score is better)
                if self._distance == Distance.COSINE:
                    if score >= self.cosine_better_than_threshold:
                        results.append({
                            "id": point.payload.get("original_id", str(point.id)),
                            "distance": score,
                            "content": point.payload.get("content", ""),
                            **{k: v for k, v in point.payload.items() 
                               if k not in ["original_id", "content"]},
                        })
                else:
                    # For distance metrics, lower is better
                    if score <= self.cosine_better_than_threshold:
                        results.append({
                            "id": point.payload.get("original_id", str(point.id)),
                            "distance": score,
                            "content": point.payload.get("content", ""),
                            **{k: v for k, v in point.payload.items() 
                               if k not in ["original_id", "content"]},
                        })

            # Return top k after filtering
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error during Qdrant query: {str(e)}")
            raise

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs."""
        try:
            # Convert original IDs to UUIDs
            point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)) for id_ in ids]
            
            if self._is_async:
                await self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=point_ids,
                    wait=True,
                )
            else:
                self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=point_ids,
                    wait=True,
                )
            
            logger.info(f"Successfully deleted {len(ids)} vectors from {self._collection_name}")
        except Exception as e:
            logger.error(f"Error deleting vectors from {self._collection_name}: {e}")
            raise

    async def delete_entity(self, entity_name: str):
        """Delete an entity by name."""
        try:
            from minirag.utils import compute_mdhash_id
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            
            logger.debug(f"Attempting to delete entity {entity_name} with ID {entity_id}")
            await self.delete([entity_id])
            logger.debug(f"Successfully deleted entity {entity_name}")
            
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")
            raise

    async def delete_relation(self, entity_name: str):
        """Delete all relations associated with an entity."""
        try:
            # Query for all points that have this entity in src_id or tgt_id
            must_conditions = []
            if "src_id" in self.meta_fields:
                must_conditions.append(
                    FieldCondition(key="src_id", match={"value": entity_name})
                )
            if "tgt_id" in self.meta_fields:
                must_conditions.append(
                    FieldCondition(key="tgt_id", match={"value": entity_name})
                )

            if not must_conditions:
                logger.warning("No src_id or tgt_id fields defined in meta_fields")
                return

            # Search for matching points
            if self._is_async:
                # Use scroll to get all matching points
                points, _ = await self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=Filter(should=must_conditions),
                    limit=10000,  # Adjust based on expected number of relations
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                points, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=Filter(should=must_conditions),
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )

            if points:
                point_ids = [str(point.id) for point in points]
                
                if self._is_async:
                    await self._client.delete(
                        collection_name=self._collection_name,
                        points_selector=point_ids,
                        wait=True,
                    )
                else:
                    self._client.delete(
                        collection_name=self._collection_name,
                        points_selector=point_ids,
                        wait=True,
                    )
                
                logger.info(f"Deleted {len(point_ids)} relations for entity {entity_name}")
            else:
                logger.debug(f"No relations found for entity {entity_name}")

        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")
            raise

    async def index_done_callback(self):
        """Called after indexing is complete. Qdrant handles persistence automatically."""
        try:
            # Optionally, you can optimize the collection here
            # self._client.update_collection(...)
            logger.debug(f"Index done callback for collection '{self._collection_name}'")
        except Exception as e:
            logger.warning(f"Error in index_done_callback: {e}")
