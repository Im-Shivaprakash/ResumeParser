import fitz
import re

def extract_data(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    links = {
        "profile_info" : {
            "linkedin" : "", 
            "mail" : "", 
            "medium" : "",
            "contact" : [],
            "location" : "" 
        }, 
        "projects" : [],
    }
    for page in doc:
        text += page.get_text()
        link_found = page.get_links()
        for link in link_found:
            if "gmail" in link["uri"]:
                links["profile_info"]["mail"] += link["uri"]
            elif "linkedin" in link["uri"]:
                links["profile_info"]["linkedin"] += link["uri"]
            elif "medium" in link["uri"]:
                links["profile_info"]["medium"] += link["uri"]
            else:
                links["projects"].append(link["uri"])
    clean_text = re.sub(r"[^a-zA-Z0-9'\"\s]", " ", text)
    for contact in re.findall(r'\b\d{10}\b', clean_text):
        links["profile_info"]["contact"].append(contact)
    doc.close()
    return clean_text, links