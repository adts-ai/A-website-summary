# Constants
import os
import requests
import json
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import openai
import streamlit as st

# Initialize environment variables and OpenAI API configuration
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_AZURE_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}
SYSTEM_PROMPT = """
You are an assistant that analyzes the contents of a website and provides a short summary, 
ignoring text that might be navigation related. Respond in.
"""

class Website:
    def __init__(self, url):
        """Initialize with the URL and scrape the content."""
        self.url = url
        self.title = None
        self.text = None
        self._scrape_content()

    def _scrape_content(self):
        """Scrape the website and parse the title and main text."""
        response = requests.get(self.url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')

        self.title = soup.title.string if soup.title else "No title found"
        self.text = self._clean_text(soup)

    def _clean_text(self, soup):
        """Remove unwanted tags and clean the text."""
        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()
            return soup.body.get_text(separator="\n", strip=True)
        return ""

    def get_user_prompt(self):
        """Generate the user prompt based on the website's content."""
        return f"You are looking at a website titled {self.title}\n" \
               f"The contents of this website are as follows:\nPlease provide a short summary of this website in. " \
               "If it includes news or announcements, summarize those too.\n\n" + self.text

    def get_messages(self):
        """Generate the messages for the OpenAI API."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.get_user_prompt()}
        ]

class WebsiteSummarizer:
    @staticmethod
    def summarize(url):
        """Summarize the website's content using OpenAI."""
        website = Website(url)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Model can be adjusted as per requirement
            messages=website.get_messages()
        )
        return response.choices[0].message.content

    @staticmethod
    def display_summary(url):
        """Fetch and display the summary of a website."""
        summary = WebsiteSummarizer.summarize(url)
        return summary

# Streamlit UI
st.title("Website Summary Generator")

url_input = st.text_input("Enter Website URL:", "https://www.cnn.com/")

if st.button("Generate Summary"):
    if url_input:
        with st.spinner("Generating summary..."):
            summary = WebsiteSummarizer.display_summary(url_input)
        st.subheader("Website Summary:")
        st.markdown(summary)
    else:
        st.warning("Please enter a valid URL.")


