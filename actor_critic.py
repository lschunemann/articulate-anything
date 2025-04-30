from typing import Callable, Any, Dict, Optional, List
from omegaconf import DictConfig
import logging
import traceback
from articulate_anything.utils.utils import (
    join_path,
    string_to_file,
    create_dir,
    Steps,
)


def error_handler(e: Exception, task, iteration: int, seed: int, cfg: DictConfig):
    """Handle errors in the actor-critic loop."""
    link_err_dir = join_path(cfg.out_dir, task,
                             f"iter_{iteration}", f"seed_{seed}")
    create_dir(link_err_dir)
    string_to_file(traceback.format_exc(),
                   join_path(link_err_dir, "error.txt"))


def default_pick_best(results):
    """
    Pick the latest iteration with the highest score and the lowest seed number
    """
    sorted_results = sorted(
        results,
        key=lambda x: (-x["iteration"], -x["feedback_score"], x["seed"]),
    )
    return sorted_results[0] if sorted_results else None


def default_load_result_func(best_result):
    return best_result


def actor_critic_loop(
    cfg: DictConfig,
    actor_func: Callable[[int, int, Dict[str, Any]], Any],
    critic_func: Callable[[Any], int],
    steps: Steps,
    pick_best_func: Optional[Callable[[list], Any]] = None,
    load_result_func: Optional[Callable[[Any], Dict[str, Any]]] = None,
    error_handler: Callable[[Exception, int, int], None] = None,
    post_process_iter: Optional[Callable[[Any], Any]] = None,
    retry_kwargs={},
    use_error_history=False  # Add this parameter
) -> Any:
    results = []
    
    # Create a deep copy of retry_kwargs to avoid modifying the original
    retry_kwargs = {k: v for k, v in retry_kwargs.items()}
    
    # Initialize global error history that persists across all iterations
    global_error_history = []
    if use_error_history and "error_history" in retry_kwargs:
        global_error_history = retry_kwargs["error_history"]
    
    # Use default pick_best function if not provided
    if pick_best_func is None:
        pick_best_func = default_pick_best
    if load_result_func is None:
        load_result_func = default_load_result_func

    for iteration in range(cfg.actor_critic.max_iter):
        for seed in range(cfg.actor_critic.num_seeds):
            try:
                # Always update retry_kwargs with the latest global error history
                if use_error_history:
                    retry_kwargs["error_history"] = global_error_history
                    logging.info(f"Iteration {iteration}, Seed {seed} - Using error history: {global_error_history}")
                
                # Run actor
                actor_result = actor_func(iteration, seed, retry_kwargs)
                
                # Run critic
                critic_result = critic_func(iteration, seed, actor_result)
                
                # Extract feedback score
                feedback_score = critic_result.get("feedback_score", -1)
                
                # Update global error history if we're using it
                if use_error_history:
                    # Extract new errors from critic result
                    new_errors = []
                    if "error_history" in critic_result:
                        new_errors = critic_result["error_history"]
                    elif critic_result.get("failure_reason", "") != "success":
                        # If critic didn't return history but found an error, create it
                        new_error = {
                            "iteration": iteration,
                            "seed": seed,
                            "error_type": critic_result.get("failure_reason", "unknown"),
                            "description": critic_result.get("improvement_suggestion", "No description provided")
                        }
                        new_errors = [new_error]
                    
                    # Add any new errors to our global history, avoiding duplicates
                    for new_error in new_errors:
                        # Check if this error already exists in global history
                        error_exists = False
                        for existing_error in global_error_history:
                            if (existing_error.get("error_type") == new_error.get("error_type") and
                                existing_error.get("description") == new_error.get("description")):
                                error_exists = True
                                break
                        
                        # Only add unique errors
                        if not error_exists:
                            global_error_history.append(new_error)
                            logging.info(f"Added new error to global history: {new_error}")
                
            except Exception as e:
                logging.error(
                    f"Failed to run actor-critic loop for iteration {iteration}, seed {seed}:")
                logging.error(traceback.format_exc())
                if error_handler:
                    error_handler(e, iteration, seed)
                feedback_score = -1
                actor_result = {}
                critic_result = {"feedback_score": -1}

            # Prepare result for this step
            step_result = {"iteration": iteration, "seed": seed}
            step_result.update(actor_result)
            step_result.update(critic_result)
            
            # Always include the global error history in the result
            if use_error_history:
                step_result["error_history"] = global_error_history.copy()

            results.append(step_result)

            best_result = pick_best_func(results)

            if post_process_iter:
                post_process_iter(best_result, cfg, steps)

            logging.info(f"At iteration {iteration}, seed {seed}, Best result so far {best_result}")
            
            # Log the current global error history
            if use_error_history:
                logging.info(f"Current global error history: {global_error_history}")

            if best_result.get('feedback_score', -1) > cfg.actor_critic.cutoff:
                logging.info(f">>> Success with result {best_result}")
                return best_result

            # Update retry_kwargs for next iteration from best result
            retry_kwargs = load_result_func(best_result)
            
            # Ensure global error history is preserved in retry_kwargs
            if use_error_history:
                retry_kwargs["error_history"] = global_error_history.copy()

            if cfg.actor_critic.conservative and feedback_score >= 0:
                logging.info(
                    f"Found successful seed {seed}, terminating iteration {iteration} early.")
                break  # Stop running more seeds for this iteration

    return pick_best_func(results)
