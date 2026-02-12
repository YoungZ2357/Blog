# Images Folder

This folder contains images for your Hugo blog posts.

Image references in your markdown files have been converted to Hugo-compatible paths:
- Obsidian syntax `![[my-image.png]]` → `![](/{repo-name}/images/my-image.png)`
- Standard markdown `![alt](my-image.png)` → `![alt](/{repo-name}/images/my-image.png)`

Images uploaded alongside your markdown files are automatically placed here.
If any referenced images were not uploaded, you can manually copy them from
your Obsidian vault into this folder.

Hugo serves files in `static/images/` at the `/{repo-name}/images/` URL path
when deployed to GitHub Pages at `https://{username}.github.io/{repo-name}/`.
