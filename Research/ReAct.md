## Adaptation to ReAct : Notes (To be integrated in Donna) <-> think, act, observe
- ReAct Approach : concatenation of (Reasoning {Thinking} + Acting ) -> llm thinking step by step before decoding next token 
- "think step by step is the key"
-  Agents {steps : Actions} -> interact with env

# Taking Actions 
JSON Agent | Code Agent | Function Calling Agents (JSON Type agent)
- Agents crucial ability : Knows when to stop

# The Stop and Parse Approach 
1) Generation in a Structured Format
2) Halting Further Generation
3) Parsing the Output

**JSON Agent / Function Calling Agents**\
Thought: {Intention underlying the users command} -> I need to check the number of layers freezed in Bert model.
{
  "action": "get_bertconfig",
  "action_input": {"config": "freeze_layers"}
}

Note : Code Agents require lesser actions to do the same 

**Observation is key; locating the action, executing {function} and gathering information, appending in the prompt at the end, re-thinking {Observing}**
