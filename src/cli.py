"""
Command-line interface for the evoprompting RAG system.
"""
import argparse
import logging
import sys

from .benchmarks.rag_bench import main as benchmark_main
from .config import Config
from .core.rag import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_command(args):
    """Setup the database schema."""
    logger.info("Setting up database...")
    Config.validate()
    rag = RAGSystem()
    rag.setup()
    rag.close()
    logger.info("Database setup complete")


def demo_command(args):
    """Run interactive demo."""
    from examples.demo_rag import main as demo_main
    demo_main()


def benchmark_command(args):
    """Run benchmark evaluation."""
    sys.argv = ['rag_bench.py']
    if args.force_reload:
        sys.argv.append('--force-reload')
    if args.max_samples:
        sys.argv.extend(['--max-samples', str(args.max_samples)])
    if args.retrieval_limit:
        sys.argv.extend(['--retrieval-limit', str(args.retrieval_limit)])

    benchmark_main()


def ask_command(args):
    """Ask a question to the RAG system."""
    Config.validate()
    rag = RAGSystem()

    try:
        result = rag.ask(args.question, limit=args.limit)
        print("\nAnswer:", result['answer'])
        print("\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source[:100]}...")
    finally:
        rag.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="evoprompting - RAG system CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evoprompting setup                          Setup database
  evoprompting demo                           Run interactive demo
  evoprompting benchmark --max-samples 100    Run benchmark
  evoprompting ask "What is RAG?"             Ask a question
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    subparsers.add_parser('setup', help='Setup database schema')

    subparsers.add_parser('demo', help='Run interactive demo')

    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmark evaluation')
    benchmark_parser.add_argument('--force-reload', action='store_true',
                                 help='Clear and reload database')
    benchmark_parser.add_argument('--max-samples', type=int, default=50,
                                 help='Maximum samples to test (default: 50)')
    benchmark_parser.add_argument('--retrieval-limit', type=int,
                                 help=f'Documents to retrieve (default: {Config.DEFAULT_RETRIEVAL_LIMIT})')

    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--limit', type=int, default=3,
                           help='Number of documents to retrieve (default: 3)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        'setup': setup_command,
        'demo': demo_command,
        'benchmark': benchmark_command,
        'ask': ask_command,
    }

    try:
        commands[args.command](args)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
