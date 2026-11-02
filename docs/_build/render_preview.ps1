$ErrorActionPreference='Stop'
$draftPath=(Resolve-Path (Join-Path $PSScriptRoot '../IBEX_User_Guide_Preview.docx')).Path
$renderDir=Join-Path $PSScriptRoot 'render'
New-Item -ItemType Directory -Force -Path $renderDir | Out-Null
$guideWord=$null
$guideDoc=$null
try {
 $guideWord=New-Object -ComObject Word.Application
 $guideWord.Visible=$false
 $guideWord.DisplayAlerts=0
 $guideDoc=$guideWord.Documents.Open($draftPath,$false,$true)
 $guideDoc.Repaginate()
 $guideDoc.ExportAsFixedFormat((Join-Path $renderDir 'IBEX_User_Guide_Preview.pdf'),17)
} finally {
 if ($guideDoc) { $guideDoc.Close(0) }
 if ($guideWord) { $guideWord.Quit() }
}
