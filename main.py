import json
import pdfplumber
import pytesseract
from PIL import Image
import io
from llamaapi import LlamaAPI

# Initialize the SDK
llama = LlamaAPI("LL-w5pm1nzp3dHPNmZw4DKC4BL3y1xE80p4jge2ZM58dOgRIs5seGlQY4RgsFAUQ2Yo")

def evaluate_paper(key_answers_pdf, student_answers_pdf):
    key_text = extract_text_from_pdf(key_answers_pdf)
    student_text = extract_text_from_pdf(student_answers_pdf)

    # Call LLM to evaluate the texts
    response = llama.run({
        "messages": [
            {"role": "user", "content": "Automatic Paper Evaluation"},
        ],
        "functions": [
            {
                "name": "compare_texts",
                "description": "Compare key answers and student answers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key_text": {
                            "type": "string",
                            "description": "Text of key answers",
                        },
                        "student_text": {
                            "type": "string",
                            "description": "Text of student answers",
                        },
                    },
                },
                "required": ["key_text", "student_text"],
            }
        ],
        "stream": False,
        "function_call": "compare_texts",
        "key_text": key_text,
        "student_text": student_text,
    })

    allocated_marks = response.json().get("allocated_marks", 0)

    # Print LLM response
    print("LLM Response:")
    print(json.dumps(response.json(), indent=2))

    return allocated_marks

def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract and append text directly from the page
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + " "
        print("----------------------")
        print(all_text)

    return all_text

# Example usage
key_answers_pdf_path = "./answer.pdf"
student_answers_pdf_path = "./right1.pdf"

# Call the evaluate_paper function
allocated_marks = evaluate_paper(key_answers_pdf_path, student_answers_pdf_path)

# Print the allocated marks
print(f"Allocated Marks: {allocated_marks}")
