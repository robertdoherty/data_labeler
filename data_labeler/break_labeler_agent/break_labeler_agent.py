# data_labeler_agent/data_labeler_agent.py
"""
DataLabelerAgent: Extract structured HVAC malfunction labels from Reddit posts.
- Loads Reddit JSON datasets (same format as reddit_research_data_* files)
- Flattens to an array of posts only
- Uses data_labeler_chains.build_data_labeler_chain() to extract key labels
- Saves labeled results to JSON
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, Union, List, Tuple
from datetime import datetime
import glob

# Ensure imports work whether run as a module or a script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Relative import when used as a package
    from .break_labeler_chains import build_data_labeler_chain
except ImportError:
    # Fallback when run directly
    from break_labeler_chains import build_data_labeler_chain

try:
    from config import DEFAULT_BREAK_MAX_CONCURRENCY
except Exception:
    DEFAULT_BREAK_MAX_CONCURRENCY = 3


class BreakLabelerAgent:
    """
    Agent for labeling Reddit posts for HVAC break/non-break and extracting fields
    """

    def __init__(self, output_dir: str = "output", log_level: int = logging.INFO):
        """
        Initialize the BreakLabelerAgent

        Args:
            output_dir: Directory for input/output files
            log_level: Logging level (default: INFO)
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging(log_level)

        # History of label runs
        self.label_history: List[Dict[str, Any]] = []
        logging.info(f"BreakLabelerAgent initialized with output_dir: {self.output_dir}")

    def _setup_logging(self, log_level: int):
        """Setup logging configuration"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, 'break_labeler_agent.log'))
            ]
        )

    def label_json_data(
        self,
        reddit_data: Union[str, Dict[str, Any]],
        output_filename: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Label Reddit JSON data for break/non-break using the LLM chain

        Args:
            reddit_data: Reddit data as JSON string or dict
            output_filename: Optional custom output filename (JSON)
            subreddits: Optional list of subreddit names to include
            max_concurrency: Optional override for concurrent LLM calls

        Returns:
            Dict with labeling results and metadata
        """
        start_time = datetime.now()
        logging.info("Starting Reddit data labeling for break/non-break...")

        try:
            # Parse reddit_data to dict
            if isinstance(reddit_data, str):
                data_obj = json.loads(reddit_data)
            else:
                data_obj = reddit_data

            # Extract posts (optionally filter) and chunk automatically to fit context
            posts = self._extract_posts(data_obj, allowed_subreddits=subreddits)
            logging.info(f"Extracted {len(posts)} posts for labeling")

            # Heuristic: keep well under Gemini 1.5 max (~1,048,575 tokens). Use ~250k chars per chunk.
            post_chunks = self._chunk_posts_by_size(posts, max_chars=250_000)
            combined_results: List[Dict[str, Any]] = []
            chain = build_data_labeler_chain()

            prepared_payloads: List[Dict[str, str]] = []
            prepared_chunks: List[Tuple[int, List[Dict[str, Any]]]] = []

            for i, chunk in enumerate(post_chunks, 1):
                chunk_json = json.dumps(chunk, ensure_ascii=False)
                logging.info(
                    "Prepared chunk %d/%d with %d posts", i, len(post_chunks), len(chunk)
                )
                prepared_payloads.append({"json_data": chunk_json})
                prepared_chunks.append((i, chunk))

            if not prepared_payloads:
                logging.info("No post chunks ready for break labeling batch execution.")
                combined_results = []
            else:
                configured_concurrency = (
                    max_concurrency
                    if isinstance(max_concurrency, int) and max_concurrency > 0
                    else DEFAULT_BREAK_MAX_CONCURRENCY
                )

                logging.info(
                    "Invoking break labeler chain for %d chunks (max_concurrency=%d)",
                    len(prepared_payloads),
                    configured_concurrency,
                )

                try:
                    outputs = chain.batch(
                        prepared_payloads,
                        config={"max_concurrency": configured_concurrency},
                    )
                except Exception as batch_exc:
                    logging.exception(
                        "Batch execution failed (max_concurrency=%d): %s; falling back to sequential processing.",
                        configured_concurrency,
                        batch_exc,
                    )
                    outputs = []
                    for idx, payload in enumerate(prepared_payloads, 1):
                        try:
                            outputs.append(chain.invoke(payload))
                        except Exception as invoke_exc:
                            logging.exception(
                                "Sequential fallback failed for chunk %d/%d: %s",
                                idx,
                                len(prepared_payloads),
                                invoke_exc,
                            )
                            outputs.append(invoke_exc)
                    logging.info(
                        "Sequential fallback complete (%d chunks)",
                        len(outputs),
                    )
                else:
                    logging.info("Batch execution complete.")

                if len(outputs) != len(prepared_chunks):
                    logging.warning(
                        "Output count (%d) does not match prepared chunk count (%d).",
                        len(outputs),
                        len(prepared_chunks),
                    )

                for idx, (chunk_index, _) in enumerate(prepared_chunks):
                    output = (
                        outputs[idx]
                        if idx < len(outputs)
                        else Exception("missing output from batch execution")
                    )
                    if isinstance(output, Exception):
                        logging.exception(
                            "Break labeler chunk %d failed: %s",
                            chunk_index,
                            output,
                        )
                        raise output

                    parsed = self._parse_chain_output(output)
                    results_arr = parsed.get("results") if isinstance(parsed, dict) else None
                    if isinstance(results_arr, list):
                        combined_results.extend(results_arr)

            labels_dict = {"results": combined_results}

            # Determine output filename
            if not output_filename:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = f"break_labeled_posts_{timestamp}.json"
            if not output_filename.endswith('.json'):
                output_filename += '.json'

            # Use output_filename as-is if it's already an absolute path or includes directory
            if os.path.isabs(output_filename) or os.path.dirname(output_filename):
                output_path = output_filename
            else:
                output_path = os.path.join(self.output_dir, output_filename)

            # Persist results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(labels_dict, f, ensure_ascii=False, indent=2)

            processing_time = (datetime.now() - start_time).total_seconds()
            result: Dict[str, Any] = {
                "success": True,
                "labels": labels_dict,
                "output_file": output_path,
                "processing_time_seconds": processing_time,
                "timestamp": start_time.isoformat(),
                "num_posts": len(posts)
            }

            self.label_history.append({
                "timestamp": start_time.isoformat(),
                "output_file": output_path,
                "processing_time": processing_time,
                "success": True,
                "num_posts": len(posts)
            })

            logging.info(f"✅ Labeling completed in {processing_time:.2f}s")
            logging.info(f"📝 Labels saved to: {output_path}")
            return result

        except Exception as e:
            error_msg = f"Labeling failed: {str(e)}"
            logging.error(error_msg, exc_info=True)

            self.label_history.append({
                "timestamp": start_time.isoformat(),
                "error": error_msg,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "success": False
            })

            return {
                "success": False,
                "error": error_msg,
                "timestamp": start_time.isoformat()
            }

    def label_from_json_file(
        self,
        json_file_path: str,
        output_filename: Optional[str] = None,
        subreddits: Optional[List[str]] = None,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Label Reddit data from a JSON file

        Args:
            json_file_path: Path to the Reddit JSON data file
            output_filename: Optional output filename for labels
            subreddits: Optional list of subreddit names to include
            max_concurrency: Optional override for concurrent LLM calls

        Returns:
            Dict with labeling results and metadata
        """
        logging.info(f"Loading Reddit data from: {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                reddit_data = json.load(f)

            # Default labeled filename derived from input
            if not output_filename:
                base_name = os.path.basename(json_file_path)
                base_name = base_name.replace("reddit_research_data_", "break_labeled_posts_")
                base_name = base_name.replace(".json", ".json")
                output_filename = base_name

            result = self.label_json_data(
                reddit_data,
                output_filename,
                subreddits=subreddits,
                max_concurrency=max_concurrency,
            )
            result["input_file"] = json_file_path
            return result
        except Exception as e:
            error_msg = f"Failed to load JSON file {json_file_path}: {str(e)}"
            logging.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "input_file": json_file_path
            }

    def label_latest_data(
        self,
        subreddits: Optional[List[str]] = None,
        max_concurrency: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find and label the most recent Reddit data file in output_dir for break/non-break

        Args:
            subreddits: Optional list of subreddit names to include
            max_concurrency: Optional override for concurrent LLM calls
        """
        logging.info("Looking for latest Reddit data file...")
        try:
            pattern = os.path.join(self.output_dir, "break_labeled_posts_*.json")
            files = glob.glob(pattern)

            if not files:
                error_msg = f"No Reddit data files found in {self.output_dir}"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}

            latest_file = max(files, key=os.path.getctime)
            logging.info(f"Found latest file: {latest_file}")
            return self.label_from_json_file(
                latest_file,
                subreddits=subreddits,
                max_concurrency=max_concurrency,
            )
        except Exception as e:
            error_msg = f"Failed to find/label latest data: {str(e)}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}

    def get_available_data_files(self) -> List[Dict[str, Any]]:
        """List available Reddit data files with basic metadata"""
        pattern = os.path.join(self.output_dir, "reddit_research_data_*.json")
        files = glob.glob(pattern)
        file_info: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                stat = os.stat(file_path)
                file_info.append({
                    "path": file_path,
                    "filename": os.path.basename(file_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                logging.warning(f"Could not get stats for {file_path}: {e}")
        file_info.sort(key=lambda x: x["created"], reverse=True)
        return file_info

    def print_status(self):
        """Print current status and available files"""
        print(f"\n{'='*60}")
        print(f"🏷️  BreakLabelerAgent Status")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"🏷️  Label Runs: {len(self.label_history)}")

        files = self.get_available_data_files()
        print(f"📋 Available Data Files: {len(files)}")
        if files:
            print("\nMost Recent Files:")
            for i, file_info in enumerate(files[:3]):
                print(f"  {i+1}. {file_info['filename']} ({file_info['size_mb']} MB)")
                print(f"     Created: {file_info['created']}")
        print(f"{'='*60}\n")

    @staticmethod
    def _extract_posts(data: Dict[str, Any], allowed_subreddits: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Flatten dataset to a list of post dicts from all subreddits
        """
        posts: List[Dict[str, Any]] = []
        if not isinstance(data, dict) or "subreddits" not in data:
            return posts
        allowed = set(s.lower() for s in (allowed_subreddits or []))
        for subreddit_name, subreddit_data in data["subreddits"].items():
            if allowed and subreddit_name.lower() not in allowed:
                continue
            for post in subreddit_data.get("posts", []):
                posts.append(post)
        return posts

    @staticmethod
    def _chunk_posts_by_size(posts: List[Dict[str, Any]], max_chars: int) -> List[List[Dict[str, Any]]]:
        """Split posts into chunks not exceeding max_chars when JSON-encoded (best-effort)."""
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        size = 0
        for post in posts:
            s = json.dumps(post, ensure_ascii=False)
            if current and size + len(s) > max_chars:
                chunks.append(current)
                current = [post]
                size = len(s)
            else:
                current.append(post)
                size += len(s)
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _parse_chain_output(chain_output: Any) -> Dict[str, Any]:
        """
        Parse the chain output into a dict. The chain is instructed to return strict JSON.
        Accept either a string or already-parsed dict.
        """
        if isinstance(chain_output, dict):
            return chain_output
        if hasattr(chain_output, "content"):
            text = chain_output.content
        else:
            text = str(chain_output)

        # Try direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            # Attempt to strip code fences or extra text by taking outermost braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except Exception:
                    pass
            # Fallback minimal structure
            return {"results": [], "raw": text}


# CLI interface when run directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BreakLabelerAgent - Label HVAC-related Reddit posts for break/non-break")
    parser.add_argument("file", nargs="?", help="Input JSON file path")
    parser.add_argument("--input", "-i", help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output labeled JSON file path")
    parser.add_argument("--latest", "-l", action="store_true", help="Label the latest Reddit data file")
    parser.add_argument("--subs", help="Comma-separated subreddit names to include (e.g., hvacadvice,skilledtrades)")
    parser.add_argument("--status", "-s", action="store_true", help="Show agent status")
    # Chunking is automatic; flags removed for simplicity
    parser.add_argument("--list", action="store_true", help="List available Reddit data files")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Optional override for concurrent LLM calls",
    )

    args = parser.parse_args()

    agent = BreakLabelerAgent(output_dir=args.output_dir)

    try:
        if args.status:
            agent.print_status()
        elif args.list:
            files = agent.get_available_data_files()
            print(f"\n📋 Available Reddit Data Files ({len(files)}):")
            print("-" * 60)
            for i, file_info in enumerate(files, 1):
                print(f"{i:2}. {file_info['filename']}")
                print(f"    Size: {file_info['size_mb']} MB")
                print(f"    Created: {file_info['created']}")
                print()
        elif args.file or args.input:
            file_path = args.file or args.input
            subs = [s.strip() for s in args.subs.split(",")] if args.subs else None
            print(f"🔍 Labeling file: {file_path}")
            result = agent.label_from_json_file(
                file_path,
                args.output,
                subreddits=subs,
                max_concurrency=args.max_concurrency,
            )
            if result.get("success"):
                print(f"✅ Labeling completed!")
                print(f"📝 Output: {result['output_file']}")
                print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
                print(f"🧾 Posts labeled: {result['num_posts']}")
            else:
                print(f"❌ Labeling failed: {result.get('error', 'Unknown error')}")
        elif args.latest:
            print("🔍 Labeling latest Reddit data file...")
            subs = [s.strip() for s in args.subs.split(",")] if args.subs else None
            result = agent.label_latest_data(
                subreddits=subs,
                max_concurrency=args.max_concurrency,
            )
            if result.get("success"):
                print(f"✅ Labeling completed!")
                print(f"📝 Output: {result['output_file']}")
                print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
                print(f"🧾 Posts labeled: {result['num_posts']}")
            else:
                print(f"❌ Labeling failed: {result.get('error', 'Unknown error')}")
        else:
            print("Please specify an action:")
            parser.print_help()
            print("\nExamples:")
            print("  python break_labeler_agent.py '99. output/reddit_research_data_2025-09-05.json'")
            print("  python break_labeler_agent.py --subs hvacadvice,HVAC '99. output/reddit_research_data_2025-09-05.json'")
            print("  python break_labeler_agent.py --latest")
            print("  python break_labeler_agent.py --status")
    except Exception as e:
        print(f"❌ Error: {e}")
        import sys as _sys
        _sys.exit(1)
