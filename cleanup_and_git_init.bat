@echo off
REM ===========================================================================
REM Cleanup temporary files and initialize git for Paper VI
REM Run this script from inside the paperVI_electromagnetism folder.
REM ===========================================================================

echo.
echo ===== Step 1: Removing temporary paper files =====
del /F /Q paper_v6old.tex 2>nul
del /F /Q paper_v7.* 2>nul
del /F /Q paper_v8.* 2>nul
del /F /Q paper_final.* 2>nul
del /F /Q paper_clean.* 2>nul
echo Temporary files removed.

echo.
echo ===== Step 2: Removing LaTeX auxiliary files =====
del /F /Q paper.aux paper.bbl paper.blg paper.log paper.out paper.toc 2>nul
del /F /Q paper.fls paper.fdb_latexmk paper.synctex.gz 2>nul
echo Aux files removed (will be regenerated on next compile).

echo.
echo ===== Step 3: Initializing git repository =====
git init
git add .gitignore
git add README.md
git add Makefile
git add paper.tex
git add references.bib
git add paper.pdf
git add code\
git add data\
git add figures\

echo.
echo ===== Step 4: First commit =====
git commit -m "Initial commit: Paper VI v11 (alpha_EM and gravitational Wilson loop)"

echo.
echo ===== Step 5: Setting up remote =====
git branch -M main
git remote add origin https://github.com/stanislasdewavrin/DDD-Paper-6-electromagnetism.git
echo Remote 'origin' added.

echo.
echo ===========================================================================
echo Repository initialized.
echo To push to GitHub, run:
echo   git push -u origin main
echo ===========================================================================
echo.
pause
