# ristisana

```cmd
python3 -m pip  install -r requirements.txt
```

## Nonogram
```
python3 nonogram.py converted.csv -s -t "Title"
```


LibreOffice Calc macro to convert colored cells to "1" and white/unset ones to "0":

```
Sub ColorToBinary
    Dim oSel As Object
    Dim oCell As Object
    Dim i As Long, j As Long
    Dim targetColor As Long

    oSel = ThisComponent.getCurrentSelection()

    ' Loop through each cell in the selection
    For i = 0 To oSel.Rows.Count - 1
        For j = 0 To oSel.Columns.Count - 1
            oCell = oSel.getCellByPosition(j, i)
            
            bg = oCell.CellBackColor

            If bg = RGB(255,255,255) Or bg = -1 Then
                oCell.String = "0"
            Else
                oCell.String = "1"
            End If
        Next j
    Next i
End Sub
```

Example file `converted.csv` has the expected data form for the python script.