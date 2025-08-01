qwen_think_template = '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n'
qwen_no_think_template = '<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'

def get_qa_prompt_reasoning(query: str, experience: list = [], previous_critical_thinking: list = []) -> str:
    reflection_analysis = ""
    for i, c in enumerate(previous_critical_thinking):
        reflection_analysis += f"Reflection {i+1}: {c}\n"

    base_str = """You are an intelligent reasoning and planning assistant designed to solve complex tasks efficiently. Your primary objective is to analyze the user's question and either provide a direct answer using your knowledge or strategically seek external help when you encounter uncertainty.

Key Principles:
- Prioritize certainty: Only provide answers when you have sufficient information
- Seek help proactively: When encountering uncertainty, immediately request external assistance rather than guessing
- Be strategic: You have a limited budget of 10 help calls, so make each one count
- Think atomically: Each sub-task should be self-contained, well-scoped, and clearly defined. You can use tools to get the information you need.  
- Avoid redundancy: Don't repeat similar tasks or request unnecessary information

You must respond in one of two formats:

If you need help or additional information, use:
<help>
Describe what kind of help you need
</help>

If you can provide the final answer, use:
<answer>
Your final answer here
</answer>

Guidelines for requests:
1. Answer Guidelines:
- Use <answer> block when you can provide the final answer based on your knowledge or available evidence
- Keep answers concise and direct - use phrases, entities, or numbers when possible
- If the question can be answered with common sense or basic knowledge, provide the answer directly
- When you have sufficient information, provide a complete answer immediately
- Focus on providing specific, actionable answers that directly address the question

2. Help Guidelines:
    - Use <help> block when you need external information or tools
    - Break down complex questions into specific, answerable queries
    - Each help request should be focused and use appropriate tools (search, calculator, code execution, etc.)
    - Request help immediately when you identify a need for external information or calculations
    - Use natural language to describe your specific requirements
    - Make one request at a time and be clear about what you need

3. Reasoning Guidelines:
- Avoid excessive internal reasoning - prioritize using appropriate tools over extended analysis
- Trust information provided in <evidence> blocks if it is valid and relevant
- Do NOT create or fabricate content within <evidence> tags - these are reserved for external information only
- Do not search for the same information again if it's already provided in <evidence> blocks
- Consider what information you already have before requesting additional help

Rules to follow:
1. Sub-tasks must be **atomic** (not composed of multiple steps) and **clearly defined**, don't have any unnecessary information.
2. **Avoid repeating similar tasks or issuing unnecessary ones — you only have 10 calls**, use calls wisely.
3. Always **consider what you already know** (including previous sub-task results) before planning a new one.
4. If the result includes a clear and valid reasoning path, you can **fully trust the answer**.
5. After each result, **update your plan** and reason about the next best step. If the subtask performs poorly, try providing a different or more specific execution method in subtask call.
6. Once the original question is fully answered, **output the final answer using**: `<answer>your final answer</answer>`


Example request for help:
<help>
I need to find the current population data for Tokyo
</help>

Example final answer:
<answer>
37.4 million
</answer>
"""

    experience_format = ""
    for i, (question, experience) in enumerate(experience):
        experience_format += f"Task {i+1}: {question}\n"
        experience_format += f"Experience {i+1}: {experience}\n\n"

    experience_str = f"""
Previous experience:
{experience_format}

Based on your previous experience, please reflect on whether any lessons learned can be applied to the current task. Consider:
- Are there similar problem types or patterns you've encountered before?
- What tools and approaches were most effective for similar tasks?
- What mistakes should be avoided based on past experience?
- Are there any specific strategies or techniques that proved successful?

Use these insights to inform your approach to the current task.
"""

    reflection_str = f"""
Previous task reflection analysis:
{reflection_analysis}

Based on the reflection analysis above, you previously completed the same task but after reflection, you realized your answer was incorrect. The reflection has identified the issues with your previous approach and reasoning. Please carefully review this analysis and redo the task with the improved understanding and corrected approach.

Key points to consider from your reflection:
- Learn from the mistakes identified in the previous attempt
- Apply the corrected reasoning approach suggested in the reflection
- Avoid repeating the same errors that led to the wrong answer
- Use the improved methodology and insights gained from the reflection

Now please redo the current task with this enhanced understanding.
"""

    task_str = f"""Current task: {query}
"""
    if reflection_analysis != "" and experience != []:
        return base_str + experience_str + reflection_str + task_str
    elif reflection_analysis != "":
        return base_str + reflection_str + task_str
    elif experience != []:
        return base_str + experience_str + task_str
    else:
        return base_str + task_str


tool_dict = {
    "search": {"description": "Search the internet for relevant information and return text results. The search results will be returned as text that can be used to answer questions or gather evidence. You need to write a search query that is suitable for web search engines. "},
    "calculator": {"description": "Generate a Python code snippet to calculate the result. The code should use exec() to execute and print the result. Include necessary imports like math if needed. Example: 'from math import sqrt\nprint(sqrt(4))'"},
    "code_execution": {"description": "Execute a Python code snippet and return the result. The code should be a valid Python code snippet that can be executed. Example: 'print(2+2)'"},
}


def get_tool_prompt(help_content: str, previous_tools: list, previous_contents: list, previous_tool_results: list) -> str:
    previous_tool_usage_context = "\n\n".join([f"Tool #{i+1}:\n" + f"tool: {tool}\n" + f"content: {content}\n" + f"result: {result}" for i, tool, content, result in zip(range(len(previous_tools)), previous_tools, previous_contents, previous_tool_results)])
    tool_description = "\n".join([f"{i+1} {tool}: {tool_dict[tool]['description']}" for i, tool in enumerate(tool_dict)])
    return f"""
    You are a helpful assistant that can answer questions by selecting and using appropriate tools.

    Your task is to:
    1. Analyze the help request and rationale
    2. Choose the most suitable tool from the available options
    3. Generate tool-specific content in the correct format
    4. Return the response in JSON format

    Available tools:
    {tool_description}

    You must respond in JSON format: 
    {{
        "tool": "tool_name",
        "content": ["tool_specific_content1", "tool_specific_content2"]
    }}

    Guidelines:
    - Choose the most appropriate tool based on the help content and rationale.
    - Ensure that the language of your output matches the language of the help content. For example, if the help content is in Chinese, your output should also be in Chinese; if the help content is in English, your output should be in English.
    - For the search tool:
        - If the help request contains multiple search intentions, decompose them into individual queries and return all queries as a list.
        - Write Google-style search queries. Effective Google search queries are typically short, specific, and focused, ranging from 2 to 5 keywords. The best queries include essential terms that clearly convey the user's intent without unnecessary words. For example, instead of asking a full question like "What is the weather like in Vienna during summer?", a more effective query would be "Vienna summer weather". Good search queries often combine a subject with a qualifier, such as location, time, or action—e.g., "Python list comprehension", "buy M1 MacBook Singapore", or "GPT-4 chain of thought". These concise, intent-rich phrases help Google deliver more accurate and relevant results.
        - You can write multiple search queries to get more information.
    - For the calculator tool, write a valid mathematical expression using basic Python libraries (math, datetime, etc.) and common libraries (numpy, pandas if needed). Avoid using specialized or uncommon libraries that may cause import errors. For example, if calculating the square root of 16, use "math.sqrt(16)" rather than "numpy.sqrt(16)" as numpy is not a basic library.
    - For the code_execution tool, use only basic Python libraries and common libraries to avoid import errors.
    - Be concise and focused in your content.
    - Ensure the content is suitable for the chosen tool.

    Example responses:
    {{
        "tool": "search",
        "content": ["capital of France", "capital of Germany"]
    }}
    {{
        "tool": "calculator", 
        "content": ["8*5+3"]
    }}
    {{
        "tool": "code_execution",
        "content": ["print('Hello, World!')"]
    }}

    Previous tool usage context:
    {previous_tool_usage_context}

    Guidelines for avoiding redundant tool calls:
    - Review previous tool usage to avoid repeating ineffective approaches
    - If a previous search query was too long or vague, simplify it
    - If a previous calculation failed, check for syntax errors or missing imports
    - If a previous code execution had issues, ensure proper formatting and dependencies
    - Learn from previous tool results to improve subsequent tool usage
    - Consider alternative approaches if previous tools didn't yield useful results

    Help content: {help_content}
    """

def get_webpage_to_reasonchain_instruction(prev_reasoning: str, help_content: str, oringinal_task: str, document: str) -> str:
    return f"""Your task is to extract factual evidence from a web page to support a larger reasoning process.

Instructions:
1. Carefully read the provided web page content.
2. Extract only objective, verifiable, and relevant facts, data, or conclusions that directly address the current search query or search intention.
3. Ignore any subjective opinions, irrelevant content, or information not directly useful for advancing the reasoning chain.
4. For each piece of helpful information, briefly explain its direct relevance to the search query or original task.
5. If you find no helpful information, reply exactly: No helpful information found
6. If helpful information is found, output it in the following format:
<evidence>
(Concise, factual information extracted from the page, with clear indication of its relevance)
</evidence>

Guidelines:
- Focus on facts, statistics, primary data, names, dates, or directly cited details.
- Do not paraphrase or interpret—extract and synthesize only factual, relevant content.
- Include information only if it fills a gap or advances the prior reasoning. Note any contradiction with previous reasoning if found.
- Do not output any content outside the <evidence> tags or the exact phrase 'No helpful information found'.

Context:
- Previous reasoning: {prev_reasoning}
- Search query: {help_content}
- Search intention: {oringinal_task}
- Web page content: {document}

Your output must be:
- Either <evidence>...</evidence> (if helpful factual content found, directly relevant and advancing the reasoning)
- Or No helpful information found (if nothing meets the above criteria)
"""


def get_summarize_experience_prompt(task_description: str, existing_experience: list = []) -> str:
    if existing_experience != []:
        experience_format = f"Task: {existing_experience[-1][0]}\n"
        experience_format += f"Experience: {existing_experience[-1][1]}\n\n"
    else:
        experience_format = "No previous experience base."
    return f"""
You are an experienced AI agent trainer tasked with analyzing task execution performance and building cumulative learning insights.

Your task is to:
1. Analyze the current task execution performance (comparing predicted vs true answer)
2. Extract generalizable insights from the reasoning process and tool usage patterns
3. Update and refine the existing experience base with new learnings
4. Focus ONLY on insights that can improve performance across diverse problem types

Current Task Analysis:
{task_description}

Existing Experience Base (400 word limit):
{experience_format}

Analysis Guidelines:

**Performance Evaluation:**
- Compare predicted answer with true answer
- Identify the root cause of success or failure in the reasoning chain
- Assess the effectiveness of tool selection and usage patterns

**Insight Extraction (MUST be generalizable across problem domains):**
- Reasoning strategies: What meta-cognitive thinking patterns led to success/failure?
- Tool usage patterns: What universal principles govern effective tool selection and usage?
- Information processing: How can external information be better integrated in any domain?
- Universal pitfalls: What cognitive biases or errors should be avoided across all tasks?
- Decision-making: What general decision frameworks were most effective?

**Experience Integration:**
- Merge new insights with existing experience, removing task-specific or redundant advice
- Prioritize universal learnings that improve success rates across diverse problem types
- Ensure ALL insights are generalizable beyond any specific domain or task type
- Maintain the 400-word limit by focusing on the most broadly applicable lessons
- If current insights conflict with existing experience, update based on evidence

**Critical Requirement: GENERALIZABILITY**
- DO NOT include task-specific details, domain knowledge, or narrow applications
- Focus on meta-strategies, universal principles, and cross-domain patterns
- Extract lessons that apply to research tasks, calculation problems, fact-finding, analysis, etc.

**Output Format:**

<updated_experience>
[Updated experience base incorporating new learnings, maximum 400 words. Focus on:
- Universal reasoning strategies that work across all problem types
- General tool usage principles and meta-patterns
- Cross-domain failure modes and avoidance strategies
- Universal decision-making guidelines
- General information integration patterns
Remove any task-specific, domain-specific, or narrowly applicable insights.]
</updated_experience>
"""


def get_reflection_prompt(question: str, reasoning: str, answer: str) -> str:
    return f"""
You are an expert evaluator with two primary tasks:
1. Determine if the answer is correct based on the reasoning process
2. If the answer is correct, determine if it can be simplified to directly answer the question

Question: {question}
Reasoning Process: {reasoning}
Final Answer: {answer}

## Task 1: Answer Correctness Evaluation

First, thoroughly evaluate the reasoning process:

**Information Source Accuracy**: 
- Are the facts, data, and evidence cited in the reasoning accurate and reliable?
- Are search results, calculations, or external information correctly interpreted?
- Are there any factual errors or misinterpretations?

**Logical Reasoning Quality**:
- Given accurate information sources, are the logical inferences reasonable and sound?
- Is the logical chain complete without gaps?
- Are the deductions properly supported by evidence?
- Is the overall approach appropriate for solving this type of problem?

**Answer Validation Criteria**:
- Mark as CORRECT only if: information sources are accurate AND logical reasoning is sound
- Mark as INCORRECT if: information sources contain errors OR logical reasoning is flawed
- Exception: If the model shows high confidence with coherent logical reasoning that appears sound, it may be acceptable even with minor information gaps

## Task 2: Answer Simplification (Only if answer is correct)

If the answer is correct, evaluate whether it directly and concisely answers the question:

**Direct Response Principle**:
- The answer should directly respond to what is asked without unnecessary elaboration
- Avoid repeating units or context already specified in the question
- Provide the most concise form that fully answers the question

**Examples of Simplification**:
- Question: "What year was Einstein born?" → Answer: "1879" (NOT "year 1879" or "Einstein was born in 1879")
- Question: "How many people attended?" → Answer: "245" (NOT "245 people")
- Question: "What is the temperature in Celsius?" → Answer: "25" (NOT "25°C" or "25 degrees Celsius")
- Question: "What is the name of the capital?" → Answer: "Paris" (NOT "the capital is Paris")

Respond in the following format:

<analysis>
answer_correctness: [correct/incorrect]
simplified_answer: [provide simplified answer, or "n/a"]
</analysis>

**Decision Rules**:
- If reasoning is invalid → answer_correctness: incorrect, simplified_answer: n/a
- If reasoning is valid and answer is correct but verbose → answer_correctness: correct, simplified_answer: [simplified answer]
- If reasoning is valid and answer is correct and already concise → answer_correctness: correct, simplified_answer: n/a
- If answer indicates uncertainty, "no answer found", "insufficient information", or similar expressions → answer_correctness: incorrect

Provide detailed reasoning for your evaluation, especially when marking reasoning as invalid or answers as incorrect.
"""
