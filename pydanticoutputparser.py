from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    task="text-generation",
   
    temperature=2,
    max_new_tokens=100
)


model=ChatHuggingFace(llm=llm)
class Person(BaseModel):
    name:str=Field(description="Name of the person")
    age:int=Field(description="Afe of the person")
    city:str=Field(description="Name of the city the person belong to")

parser=PydanticOutputParser(pydantic_object=Person)
template=PromptTemplate(
    template="Generate the name ,age and city of a  fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}


)
chain=template | model | parser
result=chain.invoke({'place':'India'})
print(result.name)

    