"""
Command-line interface for the evoprompting RAG system.
"""
import argparse
import logging
import os
import sys

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .benchmarks.rag_bench import run_benchmark_pipeline
from .config import Config
from .core.rag import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# secondary logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def setup_command(args):
    """Setup the database schema."""
    logger.info("Setting up database...")
    Config.validate()
    rag = RAGSystem()
    rag.setup()
    rag.close()
    logger.info("Database setup complete")


def benchmark_command(args):
    """Run benchmark evaluation."""
    run_benchmark_pipeline(
        subset=args.subset,
        force_reload=args.force_reload,
        max_samples=args.max_samples,
        retrieval_limit=args.retrieval_limit,
        use_llm_judge=args.use_llm_judge,
        eval_split=args.eval_split,
        use_hyde=args.use_hyde,
    )


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="evoprompting - RAG system CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            evoprompting setup                          Setup database
            evoprompting benchmark --max-samples 100    Run benchmark
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    subparsers.add_parser('setup', help='Setup database schema')

    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark evaluation')
    benchmark_parser.add_argument('--force-reload', action='store_true',
                                 help='Clear and reload database')
    benchmark_parser.add_argument('--max-samples', type=int, default=50,
                                 help='Maximum samples to test (default: 50)')
    benchmark_parser.add_argument('--retrieval-limit', type=int,
                                 help=f'Documents to retrieve (default: {Config.DEFAULT_RETRIEVAL_LIMIT})')
    benchmark_parser.add_argument('--use-llm-judge', action='store_true',
                                 help='Enable LLM-as-judge for answer evaluation')
    benchmark_parser.add_argument('--eval-split', type=str, default='test',
                                 choices=['validation', 'test'],
                                 help='Eval split: validation (for optimization) or test (final eval)')
    benchmark_parser.add_argument('--use-hyde', action='store_true',
                                 help='Use HyDE RAG instead of standard RAG')
    benchmark_parser.add_argument('--subset', type=str, default='hotpotqa',
                                 help='RAGBench subset to use (default: hotpotqa)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        'setup': setup_command,
        'benchmark': benchmark_command,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
