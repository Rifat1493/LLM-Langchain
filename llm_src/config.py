MAIN_FOLDER = 'C:/Users/Rifat1493/Desktop/LLM-Langchain/'
CACHE_FOLDER = 'C:/Users/Rifat1493/Desktop/LLM-Langchain/practice/Redis/'
COLLECTION_NAME = 'sgi_pdf'
URL = 'https://sgi.sk.ca/handbook/-/knowledge_base/drivers/driver-s-licence'
TEXT_URL = 'C:/Users/Rifat1493/Desktop/LLM-Langchain/data/fake_laptop_reviews.txt'
PDF_URL = "C:/Users/Rifat1493/Desktop/LLM-Langchain/data/drivers_handbook.pdf"
CSV_DATA = "C:/Users/Rifat1493/Desktop/LLM-Langchain/data/ds_salaries.csv"
use_4bit = True  # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit base models
bnb_4bit_quant_type = "nf4"  # Quantization type (fp4 or nf4)
use_nested_quant = False  # Activate nested quant double
