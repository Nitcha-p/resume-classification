import os
import random
import shutil
import pdfplumber
import re

DATASET_FOLDER = "PDF_Dataset"
WRONG_THRESHOLD = 0.3  # If more than 30% of resumes are misaligned, delete the folder

# Define category keywords for validation
category_keywords = {
    "INFORMATION TECHNOLOGY": ["software", "developer", "IT", "network", "database", "programming", "cloud", "cybersecurity"],
    "BUSINESS DEVELOPMENT": ["sales", "growth", "market", "strategy", "partnership", "revenue"],
    "ACCOUNTANT": ["accounting", "ledger", "financial", "tax", "audit", "bookkeeping"],
    "ADVOCATE": ["law", "attorney", "legal", "court", "litigation", "contract"],
    "ENGINEERING": ["engineer", "design", "mechanical", "electrical", "civil", "manufacturing"],
    "CHEF": ["cooking", "cuisine", "restaurant", "menu", "kitchen", "culinary"],
    "FITNESS": ["gym", "exercise", "trainer", "workout", "nutrition", "health"],
    "FINANCE": ["investment", "banking", "stocks", "financial", "capital", "economy"],
    "SALES": ["selling", "client", "customer", "deal", "negotiation", "target"],
    "AVIATION": ["pilot", "airline", "airport", "aircraft", "flight", "aerospace"],
    "HEALTHCARE": ["hospital", "doctor", "nurse", "medicine", "patient", "clinic"],
    "CONSULTANT": ["advisory", "strategy", "client", "business", "solution", "analysis"],
    "BANKING": ["bank", "loan", "credit", "investment", "finance", "account"],
    "CONSTRUCTION": ["builder", "civil", "architecture", "site", "project", "infrastructure"],
    "PUBLIC RELATIONS": ["media", "press", "communication", "branding", "advertising", "marketing"],
    "HR": ["human resources", "recruitment", "hiring", "interview", "employee", "talent"],
    "DESIGNER": ["graphic", "fashion", "interior", "product", "UI/UX", "art"],
    "ARTS": ["painting", "sculpture", "gallery", "creative", "performance", "exhibition"],
    "TEACHER": ["education", "classroom", "students", "teaching", "curriculum", "lesson"],
    "APPAREL": ["fashion", "clothing", "textile", "garment", "design", "boutique"],
    "DIGITAL MEDIA": ["social media", "content", "video", "advertising", "branding", "digital marketing"],
    "AGRICULTURE": ["farm", "agriculture", "crop", "soil", "harvest", "farming"],
    "AUTOMOBILE": ["car", "vehicle", "automobile", "engine", "mechanic", "transport"],
    "BPO": ["call center", "customer service", "support", "outsourcing", "telecalling"]
}

# Function to check if a resume contains category-related keywords
def is_resume_wrong(resume_text, category):
    category_lower = category.lower()
    
    # Check if the category name is present in the text
    if category_lower in resume_text.lower():
        return False  

    # Check if any category-related keywords are present
    keywords = category_keywords.get(category, [])
    if any(re.search(rf"\b{word}\b", resume_text, re.IGNORECASE) for word in keywords):
        return False  

    return True  # Resume does not contain relevant category-related words

# Check each folder
folders_to_delete = []
for category in os.listdir(DATASET_FOLDER):
    category_path = os.path.join(DATASET_FOLDER, category)
    
    if os.path.isdir(category_path):
        resume_files = [f for f in os.listdir(category_path) if f.endswith(".pdf")]
        total_files = len(resume_files)

        if total_files == 0:
            continue  # Skip empty folders
        
        # Adjust the number of files to check
        total_checked = min(30, int(0.3 * total_files)) if total_files > 50 else total_files
        
        wrong_count = 0
        sampled_files = random.sample(resume_files, total_checked)

        for file in sampled_files:
            pdf_path = os.path.join(category_path, file)
            with pdfplumber.open(pdf_path) as pdf:
                text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                if is_resume_wrong(text, category):
                    wrong_count += 1

        # Calculate the percentage of incorrect resumes
        error_percentage = wrong_count / total_checked
        print(f"Category: {category} - Wrong: {wrong_count}/{total_checked} ({error_percentage*100:.2f}%)")

        # If incorrect resumes exceed 30%, mark the folder for deletion
        if error_percentage > WRONG_THRESHOLD:
            folders_to_delete.append(category_path)

# Delete incorrect folders
for folder in folders_to_delete:
    shutil.rmtree(folder)
    print(f"Deleted folder: {folder}")

print("Filtering complete.")
