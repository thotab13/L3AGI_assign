from typing import List, Optional

from xagent import XAgentClient, XAgentConfiguration  # Import from XAgent framework
from xagent.agents import XAgentDialogueAgent  # Assuming XAgent provides a similar class
from xagent.schema import AIMessage, SystemMessage
from agents.agent_simulations.agent.dialogue_agent import DialogueAgent
from agents.conversational.output_parser import ConvoOutputParser
from config import Config
from memory.zep.zep_memory import ZepMemory
from services.run_log import RunLogsManager
from typings.agent import AgentWithConfigsOutput

class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        agent_with_configs: AgentWithConfigsOutput,
        system_message: SystemMessage,
        model: str,  # Assuming model is a string for XAgent configuration
        tools: List[any],
        session_id: str,
        sender_name: str,
        is_memory: bool = False,
        run_logs_manager: Optional[RunLogsManager] = None,
        **tool_kwargs,
    ) -> None:
        super().__init__(name, agent_with_configs, system_message, model)
        self.tools = tools
        self.session_id = session_id
        self.sender_name = sender_name
        self.is_memory = is_memory
        self.run_logs_manager = run_logs_manager

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """

        memory: ZepMemory

        if self.is_memory:
            memory = ZepMemory(
                session_id=self.session_id,
                url=Config.ZEP_API_URL,
                api_key=Config.ZEP_API_KEY,
                memory_key="chat_history",
                return_messages=True,
            )
            memory.human_name = self.sender_name
            memory.ai_name = self.agent_with_configs.agent.name
            memory.auto_save = False
        else:
            # Initialize memory without Zep if not using memory
            memory = None

        callbacks = []

        if self.run_logs_manager:
            # Assuming XAgent provides a callback configuration
            callbacks.append(self.run_logs_manager.get_agent_callback_handler())

        # Initialize XAgent
        agent_config = XAgentConfiguration(
            model_name=self.model,
            temperature=0,
            # Add any other necessary configuration parameters here
        )
        
        xagent_client = XAgentClient(config=agent_config)

        # Create XAgent-based agent
        agent = xagent_client.create_dialogue_agent(
            tools=self.tools,
            system_message=self.system_message.content,
            output_parser=ConvoOutputParser(),
            callbacks=callbacks,
            **tool_kwargs
        )

        prompt = "\n".join(self.message_history + [self.prefix])

        res = agent.run(input=prompt)

        if memory and self.is_memory:
            memory.save_ai_message(res)

        message = AIMessage(content=res)

        return message.content
