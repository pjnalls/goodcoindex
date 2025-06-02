import dataclasses
from logging import getLogger

from ninja import NinjaAPI

from backend.format import format_result
from backend.tinysearchengine.indexer import Document
from backend.tinysearchengine.rank import HeuristicRanker

logger = getLogger(__name__)


SCORE_THRESHOLD = 0.25


def create_router(ranker: HeuristicRanker, version: str) -> NinjaAPI:
    """
    Create a router for the search API.

    Args:
        ranker: The ranker.
        version: The version of the API.

    Returns:
        The router.
    """
    router = NinjaAPI(urls_namespace=f"search_{version}")

    @router.get("")
    def search(request, s: str):
        results = ranker.search(s, [])
        return [format_result(result) for result in results]

    @router.get("complete")
    def complete(request, q: str):
        return ranker.complete(q)

    @router.get("/raw")
    def raw(request, s: str):
        results = ranker.get_raw_results(s)
        # Convert dataclass to JSON serializable
        return {"query": s, "results": [dataclasses.asdict(result) for result in results]}

    return router
