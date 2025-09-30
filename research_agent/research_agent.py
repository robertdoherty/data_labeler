# research_agent/research_agent.py
"""
ResearchAgent class: Main interface for analyzing Reddit data using LangChain and Gemini.
- Loads Reddit JSON data from various sources
- Runs skilled trades analysis using research_chains.py
- Manages output files and provides status updates
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any, Union, List
from datetime import datetime, date
import glob

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Try relative import first (when used as module)
    from .research_chains import (
        analyze_skilled_trades_data,
        analyze_from_file,
        analyze_latest_reddit_data
    )
except ImportError:
    # Fallback to direct import (when run as script)
    from research_chains import (
        analyze_skilled_trades_data,
        analyze_from_file,
        analyze_latest_reddit_data
    )


class ResearchAgent:
    """
    Main research agent for analyzing Reddit data and generating industry insights
    """
    
    def __init__(self, output_dir: str = "output", log_level: int = logging.INFO):
        """
        Initialize the ResearchAgent
        
        Args:
            output_dir: Directory for input/output files
            log_level: Logging level (default: INFO)
        """
        self.output_dir = output_dir
        self.setup_logging(log_level)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        
        logging.info(f"ResearchAgent initialized with output_dir: {self.output_dir}")
    
    def setup_logging(self, log_level: int):
        """Setup logging configuration"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.output_dir, 'research_agent.log'))
            ]
        )
    
    def analyze_json_data(self, reddit_data: Union[str, Dict[str, Any]], 
                         output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze Reddit JSON data and generate insights
        
        Args:
            reddit_data: Reddit data as JSON string or dict
            output_filename: Optional custom output filename
            
        Returns:
            Dict with analysis results and metadata
        """
        start_time = datetime.now()
        logging.info("Starting Reddit data analysis...")
        
        try:
            # Generate output filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_filename = f"skilled_trades_analysis_{timestamp}.md"
            
            # Ensure .md extension
            if not output_filename.endswith('.md'):
                output_filename += '.md'
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Run the analysis using research_chains
            analysis_result = analyze_skilled_trades_data(reddit_data)
            
            # Save the analysis to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result metadata
            result = {
                "success": True,
                "analysis": analysis_result,
                "output_file": output_path,
                "processing_time_seconds": processing_time,
                "timestamp": start_time.isoformat(),
                "data_stats": self._get_data_stats(reddit_data)
            }
            
            # Add to history
            self.analysis_history.append({
                "timestamp": start_time.isoformat(),
                "output_file": output_path,
                "processing_time": processing_time,
                "success": True
            })
            
            logging.info(f"✅ Analysis completed in {processing_time:.2f}s")
            logging.info(f"📝 Report saved to: {output_path}")
            
            return result
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            
            # Add failed attempt to history
            self.analysis_history.append({
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
    
    def analyze_from_json_file(self, json_file_path: str, 
                              output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze Reddit data from a JSON file
        
        Args:
            json_file_path: Path to the Reddit JSON data file
            output_filename: Optional custom output filename
            
        Returns:
            Dict with analysis results and metadata
        """
        logging.info(f"Loading Reddit data from: {json_file_path}")
        
        try:
            # Load the JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                reddit_data = json.load(f)
            
            # Generate output filename based on input if not provided
            if not output_filename:
                base_name = os.path.basename(json_file_path)
                base_name = base_name.replace("reddit_research_data_", "skilled_trades_analysis_")
                base_name = base_name.replace(".json", ".md")
                output_filename = base_name
            
            # Run the analysis
            result = self.analyze_json_data(reddit_data, output_filename)
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
    
    def analyze_latest_data(self) -> Dict[str, Any]:
        """
        Find and analyze the most recent Reddit data file
        
        Returns:
            Dict with analysis results and metadata
        """
        logging.info("Looking for latest Reddit data file...")
        
        try:
            # Find all Reddit data files
            pattern = os.path.join(self.output_dir, "reddit_research_data_*.json")
            files = glob.glob(pattern)
            
            if not files:
                error_msg = f"No Reddit data files found in {self.output_dir}"
                logging.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Get the most recent file
            latest_file = max(files, key=os.path.getctime)
            logging.info(f"Found latest file: {latest_file}")
            
            # Analyze it
            return self.analyze_from_json_file(latest_file)
            
        except Exception as e:
            error_msg = f"Failed to find/analyze latest data: {str(e)}"
            logging.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_available_data_files(self) -> List[Dict[str, Any]]:
        """
        Get list of available Reddit data files with metadata
        
        Returns:
            List of dicts with file info
        """
        pattern = os.path.join(self.output_dir, "reddit_research_data_*.json")
        files = glob.glob(pattern)
        
        file_info = []
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
        
        # Sort by creation time (newest first)
        file_info.sort(key=lambda x: x["created"], reverse=True)
        return file_info
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """
        Get history of analyses performed by this agent
        
        Returns:
            List of analysis history records
        """
        return self.analysis_history.copy()
    
    def _get_data_stats(self, reddit_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract basic statistics from Reddit data
        
        Args:
            reddit_data: Reddit data as string or dict
            
        Returns:
            Dict with data statistics
        """
        try:
            if isinstance(reddit_data, str):
                data = json.loads(reddit_data)
            else:
                data = reddit_data
            
            # Extract stats from the nested structure
            stats = {
                "total_subreddits": 0,
                "total_posts": 0,
                "total_comments": 0,
                "subreddits": []
            }
            
            # Handle the nested subreddit structure
            if "subreddits" in data:
                stats["total_subreddits"] = len(data["subreddits"])
                
                for subreddit_name, subreddit_data in data["subreddits"].items():
                    posts = subreddit_data.get("posts", [])
                    posts_count = len(posts)
                    comments_count = sum(len(post.get("comments", [])) for post in posts)
                    
                    stats["total_posts"] += posts_count
                    stats["total_comments"] += comments_count
                    stats["subreddits"].append({
                        "name": subreddit_name,
                        "posts": posts_count,
                        "comments": comments_count
                    })
            
            # Also check metadata if available
            if "metadata" in data:
                metadata = data["metadata"]
                if "total_posts" in metadata:
                    stats["total_posts"] = metadata["total_posts"]
                if "total_comments" in metadata:
                    stats["total_comments"] = metadata["total_comments"]
                if "total_subreddits" in metadata:
                    stats["total_subreddits"] = metadata["total_subreddits"]
            
            return stats
            
        except Exception as e:
            logging.warning(f"Could not extract data stats: {e}")
            return {"error": str(e)}
    
    def print_status(self):
        """Print current status and available files"""
        print(f"\n{'='*60}")
        print(f"🔬 ResearchAgent Status")
        print(f"{'='*60}")
        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📊 Analyses Performed: {len(self.analysis_history)}")
        
        # Show available data files
        files = self.get_available_data_files()
        print(f"📋 Available Data Files: {len(files)}")
        
        if files:
            print("\nMost Recent Files:")
            for i, file_info in enumerate(files[:3]):  # Show top 3
                print(f"  {i+1}. {file_info['filename']} ({file_info['size_mb']} MB)")
                print(f"     Created: {file_info['created']}")
        
        # Show recent analysis history
        if self.analysis_history:
            print(f"\nRecent Analyses:")
            for i, analysis in enumerate(self.analysis_history[-3:]):  # Show last 3
                status = "✅" if analysis["success"] else "❌"
                print(f"  {status} {analysis['timestamp']} ({analysis['processing_time']:.1f}s)")
                if "output_file" in analysis:
                    print(f"     Output: {os.path.basename(analysis['output_file'])}")
        
        print(f"{'='*60}\n")
    
    def analyze_large_dataset_chunked(self, json_file_path: str, 
                                     posts_per_chunk: int = 100,
                                     max_chunks: int = 10) -> Dict[str, Any]:
        """
        Analyze large Reddit dataset by breaking it into manageable chunks
        
        Args:
            json_file_path: Path to the large Reddit JSON file
            posts_per_chunk: Number of posts per chunk (default: 100)
            max_chunks: Maximum number of chunks to process (default: 10)
            
        Returns:
            Dict with combined analysis results
        """
        logging.info(f"Starting chunked analysis of large dataset: {json_file_path}")
        logging.info(f"Chunk size: {posts_per_chunk} posts, Max chunks: {max_chunks}")
        
        try:
            # Load the full dataset
            with open(json_file_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            
            # Extract chunks
            chunks = self._create_data_chunks(full_data, posts_per_chunk, max_chunks)
            logging.info(f"Created {len(chunks)} chunks for analysis")
            
            # Process each chunk
            chunk_results = []
            combined_analysis = []
            
            for i, chunk_data in enumerate(chunks, 1):
                logging.info(f"Processing chunk {i}/{len(chunks)}...")
                
                # Analyze this chunk
                chunk_result = self.analyze_json_data(
                    chunk_data, 
                    output_filename=f"chunk_{i:02d}_analysis.md"
                )
                
                if chunk_result["success"]:
                    chunk_results.append(chunk_result)
                    combined_analysis.append(f"## Chunk {i} Analysis\n\n{chunk_result['analysis']}\n\n")
                    logging.info(f"✅ Chunk {i} completed successfully")
                else:
                    logging.error(f"❌ Chunk {i} failed: {chunk_result.get('error', 'Unknown error')}")
            
            # Combine all analyses
            if chunk_results:
                combined_text = f"# Combined Skilled Trades Analysis\n\n"
                combined_text += f"**Dataset**: {os.path.basename(json_file_path)}\n"
                combined_text += f"**Chunks Processed**: {len(chunk_results)}/{len(chunks)}\n"
                combined_text += f"**Total Posts Analyzed**: {sum(r['data_stats']['total_posts'] for r in chunk_results)}\n"
                combined_text += f"**Total Comments Analyzed**: {sum(r['data_stats']['total_comments'] for r in chunk_results)}\n\n"
                combined_text += "---\n\n"
                combined_text += "\n".join(combined_analysis)
                
                # Save combined analysis
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                combined_filename = f"combined_analysis_{timestamp}.md"
                combined_path = os.path.join(self.output_dir, combined_filename)
                
                with open(combined_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                
                logging.info(f"✅ Combined analysis saved to: {combined_path}")
                
                return {
                    "success": True,
                    "chunks_processed": len(chunk_results),
                    "total_chunks": len(chunks),
                    "combined_analysis": combined_text,
                    "combined_file": combined_path,
                    "chunk_results": chunk_results,
                    "total_posts": sum(r['data_stats']['total_posts'] for r in chunk_results),
                    "total_comments": sum(r['data_stats']['total_comments'] for r in chunk_results)
                }
            else:
                return {
                    "success": False,
                    "error": "No chunks were successfully processed"
                }
                
        except Exception as e:
            error_msg = f"Chunked analysis failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def _create_data_chunks(self, full_data: Dict[str, Any], 
                          posts_per_chunk: int, 
                          max_chunks: int) -> List[Dict[str, Any]]:
        """
        Create manageable chunks from large Reddit dataset
        
        Args:
            full_data: Complete Reddit dataset
            posts_per_chunk: Posts per chunk
            max_chunks: Maximum chunks to create
            
        Returns:
            List of chunked datasets
        """
        chunks = []
        
        if "subreddits" not in full_data:
            logging.error("Invalid data format: missing 'subreddits' key")
            return chunks
        
        # Process each subreddit
        for subreddit_name, subreddit_data in full_data["subreddits"].items():
            posts = subreddit_data.get("posts", [])
            
            if not posts:
                continue
            
            # Split posts into chunks
            for i in range(0, len(posts), posts_per_chunk):
                if len(chunks) >= max_chunks:
                    break
                
                chunk_posts = posts[i:i + posts_per_chunk]
                
                # Create chunk with same structure as original
                chunk_data = {
                    "subreddits": {
                        subreddit_name: {
                            "posts": chunk_posts
                        }
                    },
                    "metadata": {
                        "fetch_timestamp": full_data.get("metadata", {}).get("fetch_timestamp", ""),
                        "total_subreddits": 1,
                        "total_posts": len(chunk_posts),
                        "total_comments": sum(len(post.get("comments", [])) for post in chunk_posts)
                    }
                }
                
                chunks.append(chunk_data)
                
                if len(chunks) >= max_chunks:
                    break
        
        logging.info(f"Created {len(chunks)} chunks from {len(full_data['subreddits'])} subreddits")
        return chunks


# Convenience function for quick analysis
def quick_analyze(json_file_path: Optional[str] = None, output_dir: str = "output") -> Dict[str, Any]:
    """
    Quick analysis function for one-off analyses
    
    Args:
        json_file_path: Optional path to specific JSON file, if None uses latest
        output_dir: Output directory
        
    Returns:
        Analysis result dict
    """
    agent = ResearchAgent(output_dir=output_dir)
    
    if json_file_path:
        return agent.analyze_from_json_file(json_file_path)
    else:
        return agent.analyze_latest_data()


# CLI interface when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ResearchAgent - Analyze Reddit data for skilled trades insights")
    parser.add_argument("--input", "-i", help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output markdown file path")
    parser.add_argument("--latest", "-l", action="store_true", help="Analyze the latest Reddit data file")
    parser.add_argument("--chunked", "-c", action="store_true", help="Analyze large dataset in chunks")
    parser.add_argument("--posts-per-chunk", type=int, default=100, help="Posts per chunk (default: 100)")
    parser.add_argument("--max-chunks", type=int, default=10, help="Maximum chunks to process (default: 10)")
    parser.add_argument("--status", "-s", action="store_true", help="Show agent status")
    parser.add_argument("--list", action="store_true", help="List available data files")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    # Create agent
    agent = ResearchAgent(output_dir=args.output_dir)
    
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
                
        elif args.chunked:
            if args.input:
                print(f"🔍 Analyzing large dataset in chunks: {args.input}")
                result = agent.analyze_large_dataset_chunked(
                    args.input, 
                    posts_per_chunk=args.posts_per_chunk,
                    max_chunks=args.max_chunks
                )
            else:
                # Find latest file for chunked analysis
                files = agent.get_available_data_files()
                if not files:
                    print("❌ No data files found for chunked analysis")
                    sys.exit(1)
                latest_file = files[0]["path"]
                print(f"🔍 Analyzing latest large dataset in chunks: {latest_file}")
                result = agent.analyze_large_dataset_chunked(
                    latest_file,
                    posts_per_chunk=args.posts_per_chunk,
                    max_chunks=args.max_chunks
                )
            
            if result["success"]:
                print(f"✅ Chunked analysis completed!")
                print(f"📊 Chunks processed: {result['chunks_processed']}/{result['total_chunks']}")
                print(f"📝 Combined output: {result['combined_file']}")
                print(f"📈 Total posts analyzed: {result['total_posts']}")
                print(f"💬 Total comments analyzed: {result['total_comments']}")
            else:
                print(f"❌ Chunked analysis failed: {result['error']}")
                
        elif args.latest:
            print("🔍 Analyzing latest Reddit data file...")
            result = agent.analyze_latest_data()
            
            if result["success"]:
                print(f"✅ Analysis completed!")
                print(f"📊 Input: {result.get('input_file', 'N/A')}")
                print(f"📝 Output: {result['output_file']}")
                print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
                
                # Show preview
                print(f"\n{'='*50}")
                print("📋 ANALYSIS PREVIEW (first 500 chars):")
                print(f"{'='*50}")
                analysis_text = result["analysis"]
                print(analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            else:
                print(f"❌ Analysis failed: {result['error']}")
                
        elif args.input:
            print(f"🔍 Analyzing file: {args.input}")
            result = agent.analyze_from_json_file(args.input, args.output)
            
            if result["success"]:
                print(f"✅ Analysis completed!")
                print(f"📝 Output: {result['output_file']}")
                print(f"⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
            else:
                print(f"❌ Analysis failed: {result['error']}")
                
        else:
            print("Please specify an action:")
            parser.print_help()
            print("\nExamples:")
            print("  python research_agent.py --latest")
            print("  python research_agent.py --input data.json")
            print("  python research_agent.py --status")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import sys
        sys.exit(1)
