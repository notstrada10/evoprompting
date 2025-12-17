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
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Silence noisy loggers
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
    if args.use_llm_judge:
        sys.argv.append('--use-llm-judge')
    if hasattr(args, 'eval_split') and args.eval_split:
        sys.argv.extend(['--eval-split', args.eval_split])
    if hasattr(args, 'use_hyde') and args.use_hyde:
        sys.argv.append('--use-hyde')

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


def interactive_menu():
    """Display interactive menu and handle user choices."""
    print("\n" + "="*60)
    print("   EVOPROMPTING - RAG System")
    print("="*60)
    print("\n🔧 SETUP & MAINTENANCE")
    print("  1. Setup database (first-time setup)")
    print("  2. Reload knowledge base (after changing chunking params)")
    print("\n📊 EVALUATION")
    print("  3. Quick benchmark (50 samples, validation split)")
    print("  4. Full benchmark (all samples, validation split)")
    print("  5. Final test evaluation (test split, no LLM judge)")
    print("  6. Final test evaluation with LLM judge (test split)")
    print("\n🔍 OPTIMIZATION")
    print("  7. Run hyperparameter optimization (overnight)")
    print("\n💬 INTERACTIVE")
    print("  8. Ask a question")
    print("  9. Interactive demo")
    print("\n  0. Exit")
    print("="*60)

    choice = input("\nEnter your choice (0-9): ").strip()
    return choice


def handle_interactive_choice(choice):
    """Execute command based on user choice."""
    if choice == '1':
        print("\n📦 Setting up database...")
        class Args: pass
        setup_command(Args())

    elif choice == '2':
        print("\n🔄 Reloading knowledge base with current config...")
        sys.argv = ['rag_bench.py', '--force-reload', '--max-samples', '1']
        benchmark_main()

    elif choice == '3':
        print("\n📊 Running quick benchmark (50 samples, validation split)...")
        sys.argv = ['rag_bench.py', '--max-samples', '50', '--use-llm-judge', '--eval-split', 'validation']
        benchmark_main()

    elif choice == '4':
        print("\n📊 Running full benchmark (validation split)...")
        sys.argv = ['rag_bench.py', '--max-samples', '999999', '--use-llm-judge', '--eval-split', 'validation']
        benchmark_main()

    elif choice == '5':
        print("\n🎯 Running final test evaluation (no LLM judge)...")
        sys.argv = ['rag_bench.py', '--max-samples', '999999', '--eval-split', 'test']
        benchmark_main()

    elif choice == '6':
        print("\n🎯 Running final test evaluation with LLM judge...")
        sys.argv = ['rag_bench.py', '--max-samples', '999999', '--use-llm-judge', '--eval-split', 'test']
        benchmark_main()

    elif choice == '7':
        print("\n🧬 Starting hyperparameter optimization...")
        print("This will run overnight. Check optimization.log for progress.")
        import subprocess
        subprocess.run([sys.executable, 'scripts/optimize_rag_two_stage.py',
                       '--stage1-pop', '5', '--stage1-gen', '4',
                       '--stage2-pop', '6', '--stage2-gen', '5',
                       '--samples', '50'])

    elif choice == '8':
        question = input("\n💬 Enter your question: ").strip()
        if question:
            class Args:
                question = question
                limit = 5
            ask_command(Args())

    elif choice == '9':
        print("\n🎮 Starting interactive demo...")
        demo_command(None)

    elif choice == '0':
        print("\n�� Goodbye!")
        sys.exit(0)

    else:
        print("\n❌ Invalid choice. Please try again.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="evoprompting - RAG system CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evoprompting                                Interactive menu
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
    benchmark_parser.add_argument('--use-llm-judge', action='store_true',
                                 help='Enable LLM-as-judge for answer evaluation')
    benchmark_parser.add_argument('--eval-split', type=str, default='test',
                                 choices=['validation', 'test'],
                                 help='Eval split: validation (for optimization) or test (final eval)')
    benchmark_parser.add_argument('--use-hyde', action='store_true',
                                 help='Use HyDE RAG instead of standard RAG')

    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--limit', type=int, default=3,
                           help='Number of documents to retrieve (default: 3)')

    args = parser.parse_args()

    # If no command specified, show interactive menu
    if not args.command:
        while True:
            try:
                choice = interactive_menu()
                handle_interactive_choice(choice)
                if choice != '0':
                    input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
        return

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
