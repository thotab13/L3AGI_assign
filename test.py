from xagent import XAgentClient, XAgentConfiguration  # Import from XAgent framework
from langsmith import Client
from xagent_evaluation import RunEvalConfig, run_on_dataset  # Assuming XAgent has similar evaluation functions

# Initialize XAgent
def agent_factory():
    # Configure XAgent with necessary parameters
    agent_config = XAgentConfiguration(
        model_name="xagent-model",  # Replace with actual model name if different
        temperature=0,
        # Add any other necessary configuration parameters here
    )
    
    # Create XAgent client
    xagent_client = XAgentClient(config=agent_config)

    # Create and return XAgent-based agent
    return xagent_client.create_agent(
        # Add any necessary parameters for creating the agent
    )

agent = agent_factory()

# Create a Langsmith client
client = Client()

# Define evaluation configuration
eval_config = RunEvalConfig(
    evaluators=[
        "qa",
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria("conciseness"),
    ],
    input_key="input",
    eval_llm=XAgentClient(model_name="xagent-model", temperature=0.5),  # Assuming similar LLM setup
)

# Run evaluation on the dataset
chain_results = run_on_dataset(
    client,
    dataset_name="test-dataset",
    llm_or_chain_factory=agent_factory,  # Now uses XAgent
    evaluation=eval_config,
    concurrency_level=1,
    verbose=True,
)

# Output results or further processing
print(chain_results)
