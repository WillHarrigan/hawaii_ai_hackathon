<div align="center">
    <img src="./static/Kani.jpg" 
    style="width: 200px; height: auto; border-radius: 50%; border: 2px solid #ccc; align: center" />
    <h1>Kani Chatbot for Hawaiʻi’s Resilience</h1>
</div>
  
*Solution for Challenge 2 in Aloha Data:🌈🏝️AI Hackathon for Hawaiʻi’s Resilience*  
[Hackathon Details](https://datascience.hawaii.edu/ai-hackathon/)

---

## Overview

Kani Chatbot is an interactive messenger-style chatbot solution developed as a submission for Challenge 2 in the Aloha Data AI Hackathon. This project leverages modern web technologies to create a responsive chat interface that supports both text and voice inputs. The chatbot utilizes a backend service to process user messages and generate responses—ensuring an engaging and accessible user experience.

---

## Problem and Solution

Despite the abundance of publicly available climate data for Hawaiʻi, residents, students, researchers, and decision-makers struggle to access and interpret it effectively.
Current platforms are often:

•	Overly technical or fragmented

•	Difficult to navigate for non-experts

•	Lacking intuitive visualizations or summaries

•	Not designed for conversational access or personalized insights

As climate risks intensify — including rising sea levels, heatwaves, and shifting rainfall patterns — there is an urgent need to make climate information more accessible, trustworthy, and actionable for the broader public. 

Hawaiʻi faces billions in projected climate-related damages, yet limited access to understandable climate data continues to hinder informed decision-making which costs tens of millions dollars annually.

While powerful generative AI tools like ChatGPT, Gemini, Grok and LLaMA are widely available, they are not built for Hawaiʻi-specific climate data access, trust, or usability. Our Kani climate data concierge chatbot is uniquely designed to fill this critical gap by offering local relevance, verified data, and user-centered climate insights.

Our chatbot is trained and fine-tuned on local dataset such as the Hawaiʻi Climate Data Portal (HCDP). Unlike generic AIs, we understand the islands' microclimates, sea level rise zones, and adaptation challenges. Each response is backed by the real data collected from HCDP, helping users validate and trust the information — a key need not fulfilled by other AIs.

Our tool doesn't just provide data — it helps users visualize trends, download usable datasets, and explore community or policy responses. Unlike generic AIs, we’ve interviewed and surveyed users in Hawaiʻi to design the chatbot around their real needs, knowledge gaps, and desired use cases (e.g., students, residents, researchers).

---

## Features

- A clean, modern chat UI with Responsive Design.
- Built-in voice input for converting speech to text and voice reply functionality for inclusive experience.
- HCDP navigation made Easier than ever
- Easily access, process, learn, plot climate data
- LLM-powered chat
- Fully modular functionality
- Your data privacy is secured

---

## Handled Security concerns:

* Prompt Injection Attacks:
>Malicious users can craft inputs that manipulate the model’s behavior. By embedding hidden instructions, attackers might force the LLM to produce unintended or harmful outputs.

>In our case, we don't answer anything that does not fall under either of two criteria - even when one criteria is selected, the response is driven by selective data. So injection is not possible


* Data Leakage:
>Since LLMs are trained on vast datasets, there's a risk they might inadvertently reveal sensitive or private information that was part of their training data, especially if prompted in specific ways.

>In our case, our answers are strictly restricted by the tools that retrieve public data. So it is not possible to get personal info from our chat.


* Adversarial Attacks:
>Attackers can design inputs to trick the model into making errors or generating inappropriate content. These subtle, crafted inputs can lead the model to produce outputs that deviate significantly from the intended response. 

>In our case, If the query can't be answered from the extracted dtaa, we refuse to answer at all preventing itself from adversarial attacks


---

## Project Structure
```plaintext
.
├── notebooks
│   ├── this_is_for_makani.ipynb
│   ├── updated_code.ipynb
│   ├── output_files
│   └── ndvis.html
├── static
│   ├── Kani.jpg
│   ├── microphone.png
│   ├── rain.gif
│   ├── script.js
│   └── style.css
├── template
│   └── Kani 15.html
├── app.py
├── llm.py
├── README.md
├── requirements.txt
└── tokens.json
```

> **Note:** Ensure that the static assets (CSS, JS, images) are placed correctly so that the backend can serve them.

## Structure of tokens.json
```json
{
    "hcdp_api_token": "YOUR API TOKEN",
    "gmaps_api_key": "YOUR GOOGLE MAPS API KEY",
    "api_base_url": "https://api.hcdp.ikewai.org",
    "gemini_api_key": "YOUR GEMINI API KEY"
}
```

---

## Setup & Installation

1. Clone the Repository:
```bash
git clone https://github.com/WillHarrigan/hawaii_ai_hackathon.git
cd hawaii_ai_hackathon
```

2. Set Up the Environment (Optional):
```bash
python -m venv venv
source venv/bin/activate
```
> **Note:** On Windows: venv\Scripts\activate

3. Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Application:
```bash
uvicorn app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --ssl-certfile=./localhost.pem \
  --ssl-keyfile=./localhost-key.pem \
  --reload
```
> **Note:** localhost.pem and localhost-key.pem are certificates to make your website secured. Without a secured websiite, the browser may prevent you from using the voice features. 

5. [Optional] How to get .pem files?
* macOS
```bash
brew install mkcert

# if you use Firefox
brew install nss 
```

* Windows
```bash
choco install mkcert
```

* Set Up the Local CA
```bash
mkcert -install
```

* Generate Certificates for localhost
```bash
mkcert localhost
```

---
## Usage
1. Access the Chat Interface:
>Open your browser and navigate to the URL where the application is running. You will see the Kani Chatbot interface.

2. Interacting with the Chatbot:
>Text Input: Type your message in the input field and press "Enter" or click the "Send" button.

>Voice Input: Click the microphone icon to activate voice recognition. The spoken words will be converted into text and sent as a message.

>Chat Responses:The chatbot processes your message and responds. The voice reply feature will read out the chatbot's response if enabled.

---

## Customization

1. Styling:
>Modify static/style.css to adjust the appearance of the chat interface.

2. Frontend Behavior:
>Update static/script.js to change the functionality or behavior of chat interactions and voice features.

3. Backend Logic:
>Customize llm.py to change how messages are processed or integrate additional functionalities.
