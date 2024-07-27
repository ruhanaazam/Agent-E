import argparse
import asyncio
from test.tests_processor import run_tests

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Run test suite for specified range of test tasks.')

    # Add arguments
    parser.add_argument('-s', '--take_screenshots', type=bool, default=False,
                        help='Take screenshots after every operation performed (default: False)')
    parser.add_argument('-wait', '--wait_time_non_headless', type=int, default=5,
                        help='Time to wait between test tasks when running in non-headless mode (default: 5 seconds)')
    parser.add_argument('-min', '--min_task_index', type=int, default=0,
                        help='Minimum task index to start tests from (default: 0)')
    parser.add_argument('-max', '--max_task_index', type=int,
                        help='Maximum task index to end tests with, non-inclusive (default is all the tests in the file).')
    parser.add_argument('-subset', '--test_subset', type=str,
                        help='Test a subset of tasks in your test. Input should be a list of task_ids.', default=None)
    parser.add_argument('-id', '--test_results_id', type=str, default="",
                        help='A unique identifier for the test results. If not provided, a timestamp is used.')
    parser.add_argument('-config', '--test_config_file', type=str,
                        help='Path to the test configuration file. Default is "test/tasks/test.json" in the project root.')
    parser.add_argument('-validator', '--validator_type', type=str, default="text",
                        help='The maximum number of times a single tasks rerun if the run fails (e.g. due to page loading)')
    parser.add_argument('-retry', '--retry_limit', type=int, default=0,
                        help='The maximum number of times a single tasks rerun if the run fails (e.g. due to page loading)')

    # Parse the command line arguments
    args = parser.parse_args()

    # Convert string to list
    if args.test_subset:
        args.test_subset = eval(args.test_subset)

    # Run the main function with the provided or default arguments, not passing browser_manager or AutoGenWrapper will cause the test processor to create new instances of them
    asyncio.run(
        run_tests(None, None, args.min_task_index, args.max_task_index, 
                  test_results_id=args.test_results_id, 
                  test_file=args.test_config_file, 
                  take_screenshots=args.take_screenshots, 
                  wait_time_non_headless=args.wait_time_non_headless, 
                  validator_type=args.validator_type,
                  retry_limit=args.retry_limit,
                  task_subset=args.test_subset
                  )
        )
