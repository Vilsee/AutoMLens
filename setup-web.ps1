# Scaffolds the Next.js app and copies the AutoMLens UI files into it.
$ErrorActionPreference = 'Stop'
npx create-next-app@14 web --ts --tailwind --app --eslint --no-src-dir --import-alias "@/*" --use-npm
Copy-Item -Recurse -Force .\web-overlay\* .\web\
Set-Content -Path .\web\.env.local -Value 'NEXT_PUBLIC_API_URL=http://localhost:8000'
Write-Host 'Web app ready. Next: cd web; npx skills add Leonxlnx/taste-skill; npm run dev'
