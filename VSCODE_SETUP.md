# VSCode Setup Guide for Ubuntu

This guide will help you set up Visual Studio Code on Ubuntu for developing the RAG Control System Project.

## Installing VSCode on Ubuntu

### Method 1: Using Snap (Recommended)

The easiest way to install VSCode on Ubuntu is using Snap:

```bash
sudo snap install --classic code
```

### Method 2: Using apt (Debian/Ubuntu)

1. Update package index and install dependencies:
```bash
sudo apt update
sudo apt install software-properties-common apt-transport-https wget
```

2. Import Microsoft GPG key:
```bash
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
```

3. Add the VSCode repository:
```bash
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
```

4. Install VSCode:
```bash
sudo apt update
sudo apt install code
```

### Method 3: Download .deb package

1. Download the .deb package from the [official website](https://code.visualstudio.com/Download)
2. Install using dpkg:
```bash
sudo dpkg -i code_*.deb
sudo apt-get install -f  # Fix any dependency issues
```

## First Time Setup

After installing VSCode, launch it:

```bash
code
```

## Installing Recommended Extensions

When you open this project in VSCode, you'll be prompted to install recommended extensions. Click "Install All" to install:

- **Python** - Python language support
- **Pylance** - Fast Python language server
- **Black Formatter** - Python code formatter
- **Jupyter** - Jupyter notebook support
- **Prettier** - Code formatter for JSON, Markdown, etc.
- **YAML** - YAML language support
- **Markdown All in One** - Markdown editing tools
- **GitLens** - Enhanced Git capabilities
- **IntelliCode** - AI-assisted code completion
- **GitHub Copilot** - AI pair programming (requires subscription)

Alternatively, install them manually from the Extensions view (`Ctrl+Shift+X`).

## Setting Up Python Environment

### Install Python (if not already installed)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Create Virtual Environment

```bash
# Navigate to project directory
cd /path/to/RAG_ControlSystemPrj

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies (when requirements.txt is available)
pip install -r requirements.txt
```

### Configure Python Interpreter in VSCode

1. Press `Ctrl+Shift+P` to open command palette
2. Type "Python: Select Interpreter"
3. Select the interpreter from your virtual environment (`./venv/bin/python`)

## Workspace Settings

This project includes pre-configured VSCode settings in `.vscode/settings.json`:

- **Auto-save**: Files save automatically after 1 second
- **Format on Save**: Code is automatically formatted when you save
- **Python Formatting**: Uses Black formatter
- **Linting**: Pylint enabled for code quality checks
- **Tab Settings**: 4 spaces per tab
- **Trailing Whitespace**: Automatically removed on save

## Debugging

Launch configurations are provided in `.vscode/launch.json`:

1. **Python: Current File** - Debug the currently open Python file
2. **Python: Debug Tests** - Debug Python tests

To start debugging:
1. Open a Python file
2. Press `F5` or click the Run icon in the sidebar
3. Select a launch configuration

## Keyboard Shortcuts (Ubuntu)

Essential shortcuts for development:

- `Ctrl+Shift+P` - Command Palette
- `Ctrl+P` - Quick Open File
- `Ctrl+` - Toggle Terminal
- `Ctrl+B` - Toggle Sidebar
- `F5` - Start Debugging
- `Ctrl+Shift+F` - Search in Files
- `Ctrl+/` - Toggle Line Comment
- `Ctrl+D` - Select Next Occurrence
- `Ctrl+Shift+L` - Select All Occurrences
- `Alt+Up/Down` - Move Line Up/Down

## Troubleshooting

### VSCode won't start
```bash
# Try removing the GPU acceleration flag
code --disable-gpu
```

### Extension installation fails
```bash
# Check network connectivity
# Try installing from command line
code --install-extension ms-python.python
```

### Python interpreter not found
```bash
# Install Python
sudo apt install python3 python3-pip

# Verify installation
python3 --version
which python3
```

### Import errors in VSCode
- Make sure the correct Python interpreter is selected
- Check that virtual environment is activated
- Install missing packages: `pip install <package-name>`

## Additional Resources

- [VSCode Documentation](https://code.visualstudio.com/docs)
- [Python in VSCode](https://code.visualstudio.com/docs/python/python-tutorial)
- [VSCode on Linux](https://code.visualstudio.com/docs/setup/linux)
- [Ubuntu Documentation](https://ubuntu.com/tutorials)

## Contributing

When contributing to this project:

1. Make sure all recommended extensions are installed
2. Follow the code formatting settings (Black for Python)
3. Run linters before committing
4. Test your changes locally

## Support

If you encounter issues with VSCode setup on Ubuntu, please:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information about your problem
