# Azistar Main Dashboard

This project contains a web scraper and knowledge base system, along with a Flask server for text generation.

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm (Node Package Manager)
- Virtual environment (venv)

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/azistar-main-dashbaord.git
cd azistar-main-dashbaord
```

### 2. Set Up the Python Environment

1. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**:

    - **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    - **macOS/Linux**:
        ```bash
        source ./venv/bin/activate
        ```

3. **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

### 3. Set Up the Node.js Environment

1. **Install the required Node.js packages**:

    ```bash
    npm install
    ```

## Running the Application

### 1. Start the Flask Server

The Flask server handles text generation using the `google/flan-t5-large` model.

```bash
python server.py
```

### 2. Run the Web Scraper and Knowledge Base

The `wi.js` script scrapes data from URLs and queries the knowledge base.

```bash
node wi.js
```

### 3. Run the Web Scraper and Knowledge Base with OpenAI

The `wi_o.js` script scrapes data from URLs and queries the knowledge base using OpenAI.

```bash
node wi_o.js
```

## Environment Variables

Create a `.env` file in the root directory and add the following environment variables:

```
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_email_password
```

## Usage

1. **Scrape a URL**:
    - Follow the prompts to enter a URL to scrape data from.

2. **Query the Knowledge Base**:
    - Follow the prompts to enter a query and get a response based on the knowledge base.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
