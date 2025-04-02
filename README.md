# ROC Matrix Vista

A tool for interactive exploration of machine learning model performance through ROC curves, confusion matrices, and distribution visualization.

## Project Structure

```
project_root/
├── src/               # Frontend application code
│   ├── components/    # React UI components
│   ├── pages/         # Page components
│   ├── services/      # API client services
│   ├── types/         # TypeScript type definitions
│   └── lib/           # Utility functions
│
├── backend/           # Backend Python server
│   ├── src/backend/   # Main backend application
│   ├── migrations/    # Database migrations
│   └── alembic.ini    # Migration configuration
│
├── data/              # Sample data and test files
│
├── public/            # Static assets
│
└── package.json       # NPM package configuration
```

## Getting Started

### Prerequisites

- Node.js (>= 18.x)
- Python (>= 3.10)
- Poetry or pip for Python dependencies

### Frontend Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies using Poetry
poetry install
# OR using pip
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run migrations
cd backend
alembic upgrade head

# Start the backend server
python -m src.backend.main
```

## Features

- Interactive ROC curve with adjustable decision threshold
- Visual exploration of model performance metrics
- Prediction score distribution visualization
- Confusion matrix with real-time updates
- Model performance metrics at different decision thresholds

## Development

See the [frontend README](src/README.md) and [backend README](backend/README.md) for more detailed information about each part of the application.

## Project info

**URL**: https://lovable.dev/projects/2928b1e9-7e8a-4405-81a2-ea4e5153fc1b

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/2928b1e9-7e8a-4405-81a2-ea4e5153fc1b) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with .

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/2928b1e9-7e8a-4405-81a2-ea4e5153fc1b) and click on Share -> Publish.

## I want to use a custom domain - is that possible?

We don't support custom domains (yet). If you want to deploy your project under your own domain then we recommend using Netlify. Visit our docs for more details: [Custom domains](https://docs.lovable.dev/tips-tricks/custom-domain/)
