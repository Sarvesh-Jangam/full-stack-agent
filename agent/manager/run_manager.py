import asyncio
from agent.manager.agent import ManagerRequest, process_manager_request

async def run_manager():
    request = ManagerRequest(
        action="start_workflow",
        user_input="Create a modern ecommerce site with authentication and deployment."
    )
    response = await process_manager_request(request)
    print(response)

asyncio.run(run_manager())
