#!/bin/bash

# Check if tmux is installed, if not, install it
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    sudo apt-get update
    sudo apt-get install -y tmux
fi

# Check if git is installed, if not, install it
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    sudo apt-get update
    sudo apt-get install -y git
fi
# Set default editor for git to "vim"
git config --global core.editor "vim"
git config --global credential.helper store


# Function to copy files if they exist
copy_file() {
    if [ -f "$1" ]; then
        cp "$1" "$2"
        echo "Copied $1 to $2"
    else
        echo "File $1 not found. Skipping..."
    fi
}

# Check if the user's home directory exists
if [ ! -d "$HOME" ]; then
    echo "Error: Home directory not found."
    exit 1
fi

# Copy .bashrc
copy_file "bashrc" "$HOME/.bashrc"

# Copy .tmux.conf
copy_file "tmux.conf" "$HOME/.tmux.conf"

# Copy .vimrc
copy_file "vimrc" "$HOME/.vimrc"

