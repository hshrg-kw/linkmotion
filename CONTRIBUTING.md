# Contribution Guide
Thank you for your interest in contributing to our project! We welcome all contributions, from bug reports to new features. To ensure a smooth process for everyone, please read and follow these guidelines.


# Code of Conduct
To foster an open and welcoming environment, we have adopted a Code of Conduct. We expect all contributors to adhere to it. Please take a moment to read it before you start.


# How to Contribute
We encourage you to contribute! Here are the ways you can help:

- Reporting Bugs: If you find a bug, please create an issue in our issue tracker.

- Suggesting Enhancements: If you have an idea for a new feature or an improvement, create an issue to discuss it.

- Submitting Pull Requests: If you want to fix a bug or implement a feature, please submit a Pull Request.

Before starting work, please search the existing issues to see if your idea or bug has already been discussed.


# Getting Started: Setting Up Your Development Environment
To get started with development, please follow these steps:

1. **Fork the repository** on GitHub.

2. **Clone your fork** to your local machine:
    ```Bash
    git clone https://github.com/YOUR_USERNAME/PROJECT_NAME.git
    cd PROJECT_NAME
    ```

3. **Install dependencies**. We use uv for package management.
    ```Bash
    # Install project dependencies
    uv sync
    ```

4. **Run the tests** and confirm passing all of them.
    ```Bash
    uv run pytest
    ```

Now you are ready to start making changes!


# Development Workflow (Branching Strategy)
This project follows the GitHub Flow model. It's a simple and effective workflow.

1. Create a new branch from the main branch. Please use a descriptive name.

    - Use the following prefixes for branch names:

        - feature/: For new features (e.g., feature/user-profile-page)

        - fix/: For bug fixes (e.g., fix/login-validation-error)

        - chore/: For maintenance tasks (e.g., chore/update-dependencies)
            ```Bash
            # Make sure you are on the main branch and have the latest changes
            git checkout main
            git pull origin main

            # Create your new branch
            git checkout -b feature/your-new-feature
            ```

2. Make your changes and commit them. Make sure to follow the following Commit Message Guidelines.

3. Push your branch to your fork on GitHub:
    ```Bash
    git push origin feature/your-new-feature
    ```
    
4. Create a Pull Request (PR) from your fork to the main branch of the original repository.

5. Wait for your PR to be reviewed. Respond to any feedback or requested changes.

6. Once your PR is approved and passes all checks, it will be merged into main.


# Coding Style Guide
To maintain a consistent codebase, we enforce a coding style.

- Linting and Formatting: We use ESLint for linting and Prettier for code formatting.

- Before Committing: Please run the linter and formatter to ensure your code adheres to our style.
    ```Bash
    # Run the linter to catch any issues
    uv run ruff check

    # Automatically format your code
    uv run ruff format
    ```

- Pull Requests with linting errors will not be merged.


# Commit Message Guidelines
We follow the Conventional Commits specification. This helps us automate changelog generation and makes the project history more readable.

## Format
Each commit message consists of a header, a body, and a footer.
```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```
- Type: Must be one of the following:

    - feat: A new feature.

    - fix: A bug fix.

    - docs: Documentation only changes.

    - style: Changes that do not affect the meaning of the code (white-space, formatting, etc).

    - refactor: A code change that neither fixes a bug nor adds a feature.

    - test: Adding missing tests or correcting existing tests.

    - chore: Changes to the build process or auxiliary tools and libraries.

- Scope (optional): A noun specifying the section of the codebase affected.

- Subject: A concise description of the change. Use the imperative, present tense: "add" not "added" nor "adds".

## Examples
### Example of a feature commit:
```
feat(api): add endpoint for user authentication
```
### Example of a fix commit with a body and footer:
```
fix(auth): correct password reset email dispatch

The email service was misconfigured, causing password reset emails
to fail. This commit updates the configuration and adds a
corresponding integration test.

Closes #123
```


# Submitting a Pull Request (PR)
- Give your PR a clear and descriptive title, following the commit message format (e.g., feat(auth): Implement social login).

- In the PR description, explain the "what" and "why" of your changes. Link to any relevant issues.

- Ensure that your PR passes all automated checks (e.g., linting, tests).

- If your PR is a work in progress, please create it as a Draft Pull Request.

Thank you for your contribution!