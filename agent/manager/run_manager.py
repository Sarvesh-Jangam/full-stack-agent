import asyncio
import logging
from agent.manager.agent import ManagerRequest, process_manager_request, default_runner

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Main function to initialize, run, and complete a workflow with the manager agent.
    """
    user_id = "default_user"
    session_id = "default_session"
    max_iterations = 10  # Safety break to prevent infinite loops
    current_iteration = 0

    try:
        # 1. Create the session
        logging.info("--- Creating Session ---")
        await default_runner.create_session(user_id=user_id, session_id=session_id)
        logging.info(f"Session '{session_id}' created for user '{user_id}'.")

        # 2. Start the workflow
        logging.info("\n--- Starting Workflow ---")
        initial_request = ManagerRequest(
            action="start_workflow",
            payload={"task_description": "Create a full-stack web application with a user database and a simple UI."}
        )
        response = await process_manager_request(
            request=initial_request,
            user_id=user_id,
            session_id=session_id
        )
        logging.info(f"Workflow started. Response: {response.model_dump_json(indent=2)}")

        # 3. Loop through the workflow until it's complete
        while response.status != "completed" and current_iteration < max_iterations:
            current_iteration += 1
            logging.info(f"\n--- Workflow Iteration {current_iteration} (Status: {response.status}) ---")

            # Determine the next action from the response
            if not response.next_actions:
                logging.error("Workflow stalled: No next actions provided.")
                break
            
            # For this simulation, we'll just pick the first suggested action
            next_action = response.next_actions[0]
            logging.info(f"Executing next action: '{next_action}'")

            # Create the next request
            next_request = ManagerRequest(action=next_action, payload={})
            
            response = await process_manager_request(
                request=next_request,
                user_id=user_id,
                session_id=session_id
            )
            logging.info(f"Iteration response: {response.model_dump_json(indent=2)}")
            
            # Small delay to make the process easier to follow
            await asyncio.sleep(1)

        # 4. Final outcome
        logging.info("\n--- Workflow Finished ---")
        if response.status == "completed":
            logging.info("Workflow completed successfully!")
        else:
            logging.warning(f"Workflow ended with status: '{response.status}'.")
        
        logging.info(f"Final artifacts: {response.artifacts}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())