import asyncio
import json
import os
import time
from test.evaluators import evaluator_router
from test.test_utils import load_config
from test.test_utils import task_config_validator
from typing import Any

import ae.core.playwright_manager as browserManager
import nltk  # type: ignore
from ae.config import PROJECT_TEST_ROOT
from ae.core.autogen_wrapper import AutogenWrapper
from ae.core.playwright_manager import PlaywrightManager
from ae.utils.logger import logger
from autogen.agentchat.chat import ChatResult  # type: ignore
from playwright.async_api import Page
from tabulate import tabulate
from termcolor import colored

nltk.download('punkt') # type: ignore

TEST_TASKS = os.path.join(PROJECT_TEST_ROOT, 'tasks')
TEST_LOGS = os.path.join(PROJECT_TEST_ROOT, 'logs')
TEST_RESULTS = os.path.join(PROJECT_TEST_ROOT, 'results')

last_agent_response = ""

def check_test_folders():
    if not os.path.exists(TEST_LOGS):
        os.makedirs(TEST_LOGS)
        logger.info(f"Created log folder at: {TEST_LOGS}")

    if not os.path.exists(TEST_RESULTS):
        os.makedirs(TEST_RESULTS)
        logger.info(f"Created scores folder at: {TEST_RESULTS}")


def dump_log(task_id: str, messages_str_keys: dict[str, str]):
    file_name = os.path.join(TEST_LOGS, f'execution_logs_{task_id}.json')
    with open(file_name, 'w',  encoding='utf-8') as f:
            json.dump(messages_str_keys, f, ensure_ascii=False, indent=4)


def save_test_results(test_results: list[dict[str, str | int | float | None]], test_results_id: str):
    file_name = os.path.join(TEST_RESULTS, f'test_results_{test_results_id}.json')
    with open(file_name, 'w',  encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    logger.info(f"Test results dumped to: {file_name}")


def extract_last_response(messages: list[dict[str, Any]]) -> str:
    """Extract the last response message from chat history."""
    # Iterate over the messages in reverse order
    for message in reversed(messages):
        if '##TERMINATE##' in message.get('content', ''):
            return message['content'].replace("##TERMINATE##", "").strip()
    return ""


def print_progress_bar(current: int, total: int, bar_length: int = 50) -> None:
    """
    Prints a progress bar to the console.

    Parameters:
    - current (int): The current progress of the task.
    - total (int): The total number of tasks to complete.
    - bar_length (int): The character length of the progress bar (default is 50).

    This function dynamically updates a single line in the console to reflect current progress.

    """
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print(f'\rProgress: [{arrow}{spaces}] {current}/{total} ({percent:.2f}%)', end='')


def print_test_result(task_result: dict[str, str | int | float | None], index: int, total: int) -> None:
    """
    Prints the result of a single test task in a tabulated format.

    Parameters:
    - task_result (dict): A dictionary containing the task's evaluation results, including task ID, intent, score, and total command time.
    - index (int): The current index of the test in the sequence of all tests being run.
    - total (int): The total number of tests to be run.

    The function determines the test status (Pass/Fail) based on the 'score' key in task_result and prints the result with colored status.

    """
    status = 'Pass' if task_result['score'] == 1 else 'Fail'
    color = 'green' if status == 'Pass' else 'red'
    print(f"\n\nTest {index}/{total} Results: {task_result}")

    #Cost computation is not available in Autogen when using Nested chat agents
    cost="NA" 
    total_cost = "NA"
    total_tokens = "NA"
    if "compute_cost" in task_result:
        cost = task_result["compute_cost"]
        total_cost = round(cost.get("cost", -1), 4) # type: ignore
        total_tokens = cost.get("total_tokens", -1) # type: ignore
    result_table = [  # type: ignore
        ['Test Index', 'Task ID', 'Intent', 'Status', 'Time Taken (s)', 'Total Tokens', 'Total Cost ($)'],
        [index, task_result['task_id'], task_result['intent'], colored(status, color), round(task_result['tct'], 2), total_tokens, total_cost]  # type: ignore
    ]
    print('\n' + tabulate(result_table, headers='firstrow', tablefmt='grid')) # type: ignore

def get_command_exec_cost(command_exec_result: ChatResult):
    cost = command_exec_result.cost # type: ignore
    usage: dict[str, Any] = None
    if "usage_including_cached_inference" in cost:
        usage: dict[str, Any] = cost["usage_including_cached_inference"]
    elif "usage_excluding_cached_inference" in cost:
        usage: dict[str, Any] = cost["usage_excluding_cached_inference"]
    else:
        raise ValueError("Cost not found in the command execution result.")
    print("Usage: ", usage)
    output: dict[str, Any] = {}
    for key in usage.keys():
        if isinstance(usage[key], dict) and "prompt_tokens" in usage[key]:
            output["cost"] = usage[key]["cost"]
            output["prompt_tokens"] = usage[key]["prompt_tokens"]
            output["completion_tokens"] = usage[key]["completion_tokens"]
            output["total_tokens"] = usage[key]["total_tokens"]
    return output


async def execute_single_task(task_config: dict[str, Any], browser_manager: PlaywrightManager, ag: AutogenWrapper, page: Page) -> dict[str, Any]:
    """
    Executes a single test task based on a specified task configuration and evaluates its performance.

    Parameters:
    - task_config (dict): The task configuration dictionary containing all necessary parameters for the task.
    - browser_manager (PlaywrightManager): The manager handling browser interactions, responsible for page navigation and control.
    - ag (AutogenWrapper): The automation generator wrapper that processes commands and interacts with the web page.
    - page (Page): The Playwright page object representing the browser tab where the task is executed.

    Returns:
    - dict: A dictionary containing the task's evaluation results, including task ID, intent, score, total command time (tct),
            the last statement from the chat agent, and the last URL accessed during the task.
    """
    command = ""
    start_url = None
    task_id = None

    task_config_validator(task_config)

    command: str = task_config.get('intent', "")
    task_id = task_config.get('task_id')
    start_url = task_config.get('start_url')
    logger.info(f"Intent: {command}, Task ID: {task_id}")

    if start_url:
        await page.goto(start_url, wait_until='load', timeout=30000)

    start_time = time.time()
    current_url = await browser_manager.get_current_url()
    command_exec_result = await ag.process_command(command, current_url)
    await page.wait_for_selector('body', timeout=60000)
    try:
        await page.screenshot(path=f"./test/screenshots/task_{task_id}.png") #try to take a screenshot
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")

    end_time = time.time()
    
    logger.info(f"Command \"{command}\" took: {round(end_time - start_time, 2)} seconds.")
    logger.info(f"Task {task_id} completed.")

    messages = ag.agents_map["planner_agent"].chat_messages # type: ignore
    messages_str_keys = {str(key): value for key, value in messages.items()} # type: ignore
    
    agent_key = list(messages.keys())[0] # type: ignore

    last_agent_response = extract_last_response(messages[agent_key]) # type: ignore
    dump_log(str(task_id), messages_str_keys)
    evaluator = evaluator_router(task_config)

    cdp_session = await page.context.new_cdp_session(page)
    score = await evaluator(
        task_config=task_config,
        page=page,
        client=cdp_session,
        answer=last_agent_response,
    )
    try:
        command_cost = get_command_exec_cost(command_exec_result)
        print(f"Command cost: {command_cost}")
    except Exception as e:
        logger.error(f"Error getting command cost: {e}")
        command_cost = {"cost": -1, "total_tokens": -1}
    return {
        "task_id": task_id,
        "start_url": start_url,
        "intent": str(command),
        "score": score,
        "tct": end_time - start_time,
        "last_statement": last_agent_response,
        "last_url": page.url,
        "compute_cost": command_cost
    }


async def run_tests(ag: AutogenWrapper, browser_manager: PlaywrightManager, min_task_index: int, max_task_index: int,
               test_file: str="", test_results_id: str = "", wait_time_non_headless: int=5) -> list[dict[str, Any]]:
    """
    Runs a specified range of test tasks using Playwright for browser interactions and AutogenWrapper for task automation.
    It initializes necessary components, processes each task, handles exceptions, and compiles test results into a structured list.

    Parameters:
    - ag (AutogenWrapper): The AutoGen wrapper that processes commands.
    - browser_manager (PlaywrightManager): The manager handling browser interactions, responsible for page navigation and control.
    - min_task_index (int): The index of the first test task to execute (inclusive).
    - max_task_index (int): The index of the last test task to execute (non-inclusive).
    - test_file (str): Path to the file containing the test configurations. If not provided, defaults to a predetermined file path.
    - test_results_id (str): A unique identifier for the session of test results. Defaults to a timestamp if not provided.
    - wait_time_non_headless (int): Time to wait between tasks when running in non-headless mode, useful for live monitoring or debugging.

    Returns:
    - list[dict[str, Any]]: A list of dictionaries, each containing the results from executing a test task. Results include task ID, intent, score, total command time, etc.

    This function also manages logging and saving of test results, updates the progress bar to reflect test execution status, and prints a detailed summary report at the end of the testing session.
    """
    if not test_file or test_file == "":
        test_file = os.path.join(TEST_TASKS, 'test.json')

    logger.info(f"Loading test configurations from: {test_file}")

    test_configurations = load_config(test_file)

    if not test_results_id or test_results_id == "":
        test_results_id = str(int(time.time()))

    check_test_folders()
    test_results: list[dict[str, str | int | float | None]] = []

    if not ag:
        ag = await AutogenWrapper.create()

    if not browser_manager:
        browser_manager = browserManager.PlaywrightManager(headless=False)
        await browser_manager.async_initialize()

    page=await browser_manager.get_current_page()
    test_results = []
    max_task_index = len(test_configurations) if not max_task_index else max_task_index
    total_tests = max_task_index - min_task_index

    for index, task_config in enumerate(test_configurations[min_task_index:max_task_index], start=min_task_index):
        print_progress_bar(index - min_task_index, total_tests)
        task_result = await execute_single_task(task_config, browser_manager, ag, page)
        test_results.append(task_result)
        save_test_results(test_results, test_results_id)
        print_test_result(task_result, index + 1, total_tests)
        await browser_manager.close_except_specified_tab(page) #cleanup pages that are not the one we opened here

        if not browser_manager.isheadless: #no need to wait if we are running headless
            await asyncio.sleep(wait_time_non_headless)  # give time for switching between tasks in case there is a human observer

    print_progress_bar(total_tests, total_tests)  # Complete the progress bar
    print('\n\nAll tests completed.')

    # Aggregate and print individual test results
    print("\nDetailed Test Results:")
    detailed_results_table = [['Test Index', 'Task ID', 'Intent', 'Status', 'Time Taken (s)', 'Total Tokens', 'Total Cost ($)']]
    for idx, result in enumerate(test_results, 1):
        status = 'Pass' if result['score'] == 1 else 'Fail'
        color = 'green' if status == 'Pass' else 'red'

        cost = result["compute_cost"]
        total_cost = round(cost.get("cost", -1), 4) # type: ignore
        total_tokens = cost.get("total_tokens", -1) # type: ignore

        detailed_results_table.append([
            idx, result['task_id'], result['intent'], colored(status, color), round(result['tct'], 2), # type: ignore
            total_tokens, total_cost
        ])
    print(tabulate(detailed_results_table, headers='firstrow', tablefmt='grid'))

    # Summary report

    # Calculate aggregated cost and token totals
    total_cost = sum(result["compute_cost"].get("cost", 0) for result in test_results) # type: ignore
    total_tokens = sum(result["compute_cost"].get("total_tokens", 0) for result in test_results) # type: ignore

    passed_tests = [result for result in test_results if result['score'] == 1]
    summary_table = [ # type: ignore
        ['Total Tests', 'Passed', 'Failed', 'Average Time Taken (s)', 'Total Time Taken (s)', 'Total Tokens', 'Total Cost ($)'],
        [total_tests, len(passed_tests), total_tests - len(passed_tests),
         round(sum(test['tct'] for test in test_results) / total_tests, 2), # type: ignore
         round(sum(test['tct'] for test in test_results), 2),  # type: ignore
         total_tokens, total_cost]
    ]

    print('\nSummary Report:')
    print(tabulate(summary_table, headers='firstrow', tablefmt='grid')) # type: ignore

    return test_results
