# First upload

1. The project is licensed under GNU GPL version 3 or later (`GPL-3.0-or-later`). Keep both `LICENSE` and `NOTICE` in the repository. Academic and commercial use are permitted under these terms.
2. Create a new empty repository on GitHub named `IBEX-Machine-Learning-Studio` (or your preferred name). Do not initialize it with a second README or .gitignore.
3. In GitHub Desktop, choose **File → Add local repository**, select this folder, and create a repository here if prompted. Commit the prepared files, then use **Publish repository**. Alternatively use the commands below.
4. Set visibility to Public when ready to share. Repository creation and upload have not been performed by the folder preparation.
5. Publish the self-contained EXE as a separate GitHub Release asset using the instructions in README.md. Tag the matching source version and provide its complete corresponding source and build scripts alongside the binary release, together with `LICENSE` and `NOTICE`, in accordance with GPL requirements.

Command-line alternative (replace the example URL with your actual repository URL):

```powershell
git init -b main
git add .
git commit -m "Initial source release of IBEX Machine Learning Studio"
git remote add origin https://github.com/YOUR-USERNAME/IBEX-Machine-Learning-Studio.git
git push -u origin main
```

This folder contains the standalone ML application. OptStudio, the Microsoft Store packaging project, signing credentials, local editor settings, compiled programs and temporary test output are excluded.
