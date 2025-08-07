import asyncio
from agent.manager.agent import ManagerRequest, process_manager_request, default_runner

async def main():
    """
    Main function to initialize and run the manager agent.
    """
    # Define a default user and session ID
    user_id = "default_user"
    session_id = "default_session"

    # --- FIX: Create the session before using it ---
    # This ensures that the session exists when the agent tries to access it.
    await default_runner.create_session(user_id=user_id, session_id=session_id)

    # Now, create the initial request to start the workflow
    initial_request = ManagerRequest(
        action="start_workflow",
        payload={"task_description": "Create a full-stack web application."}
    )

    # Process the request
    response = await process_manager_request(
        request=initial_request,
        user_id=user_id,
        session_id=session_id
    )

    # Print the final response
    print(response)

if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main())

