import json
from typing import Any, Dict, List, Optional, Union


def _load(data_or_path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(data_or_path, str):
        with open(data_or_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return data_or_path


def build_enriched_json(
    labeled_results: Union[str, Dict[str, Any]],
    reddit_data: Union[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    labeled = _load(labeled_results)
    data = _load(reddit_data)

    post_index: Dict[str, Dict[str, Any]] = {}
    for _sr, sr_data in (data.get("subreddits", {}) or {}).items():
        for post in sr_data.get("posts", []) or []:
            pid = post.get("post_id")
            if pid:
                post_index[pid] = post

    combined: List[Dict[str, Any]] = []
    for item in labeled.get("results", []) or []:
        pid = item.get("id")
        meta = dict(post_index.get(pid, {}) or {})
        if "post_id" in meta:
            meta["id"] = meta.get("post_id")
            meta.pop("post_id", None)
        elif "id" not in meta and pid:
            meta["id"] = pid
        meta.pop("comments", None)
        merged = {**meta, **item}
        combined.append(merged)

    result = {"results": combined}
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


