open System
open System.IO
open DLKcatService

[<Literal>]
let pythonPath = @"C:\Users\Anyee\source\repos\FsTest\NetPython\python_embedded"
[<Literal>]
let dlkcatPath = @"C:\Users\Anyee\source\repos\FsTest\NetPython\DLKcat"

let data =
    { name = "L-Arginine"
      smlies = "C(CC(C(=O)O)N)CN=C(N)N"
      sequence =
        "MSLGIRYLALLPLFVITACQQPVNYNPPATQVAQVQPAIVNNSWIEISRSALDFNVKKVQSLLGKQSSLCAVLKGDAYGHDLSLVAPIMIENNVKCIGVTNNQELKEVRDLGFKGRLMRVRNATEQEMAQATNYNVEELIGDLDMAKRLDAIAKQQNKVIPIHLALNSGGMSRNGLEVDNKSGLEKAKQISQLANLKVVGIMSHYPEEDANKVREDLARFKQQSQQVLEVMGLERNNVTLHMANTFATITVPESWLDMVRVGGIFYGDTIASTDYKRVMTFKSNIASINYYPKGNTVGYDRTYTLKRDSVLANIPVGYADGYRRVFSNAGHALIAGQRVPVLGKTSMNTVIVDITSLNNIKPGDEVVFFGKQGNSEITAEEIEDISGALFTEMSILWGATNQRVLVD" }
let data2 =
    { name = "Catechol"
      smlies = "C1=CC=C(C(=C1)O)O"
      sequence =
        "MVHVRKNHLTMTAEEKRRFVHAVLEIKRRGIYDRFVKLHIQINSTDYLDKETGKRLGHVNPGFLPWHRQYLLKFEQALQKVDPRVTLPYWDWTTDHGENSPLWSDTFMGGNGRPGDRRVMTGPFARRNGWKLNISVIPEGPEDPALNGNYTHDDRDYLVRDFGTLTPDLPTPQELEQTLDLTVYDCPPWNHTSGGTPPYESFRNHLEGYTKFAWEPRLGKLHGAAHVWTGGHMMYIGSPNDPVFFLNHCMIDRCWALWQARHPDVPHYLPTVPTQDVPDLNTPLGPWHTKTPADLLDHTRFYTYDQ" }

let data3 =
    { name = "N-(5-Phospho-D-ribosyl)anthranilate"
      smlies = "C1=CC=C(C(=C1)C(=O)O)NC2C(C(C(O2)COP(=O)(O)O)O)O"
      sequence =
        "MSVINFTGSSGPLVKVCGLQSTEAAECALDSDADLLGIICVPNRKRTIDPVIARKISSLVKAYKNSSGTPKYLVGVFRNQPKEDVLALVNDYGIDIVQLHGDESWQEYQEFLGLPVIKRLVFPKDCNILLSAASQKPHSFIPLFDSEAGGTGELLDWNSISDWVGRQESPESLHFMLAGGLTPENVGDALRLNGVIGVDVSGGVETNGVKDSNKIANFVKNAKK" }

[<EntryPoint>]
let main argv =
    let a = new DLKcatPrediction(pythonPath, dlkcatPath)
    a.predictForInput data |> printfn "Result: %s"
    a.predictForInput data2 |> printfn "Result: %s"
    a.predictForInput data3 |> printfn "Result: %s"

    a.shutdown ()

    0








// let pyObj (value: obj) : PyObject =
//     match value with
//     | :? string as s -> new PyString(s) :> PyObject
//     | :? int as i -> new PyInt(i) :> PyObject
//     | _ -> failwith "Unsupported type"

// let initializePython (pythonPath: string) =
//     Runtime.PythonDLL <- Path.Combine(pythonPath, "python37.dll")
//     PythonEngine.PythonPath <-
//         String.Join(
//             string Path.PathSeparator,
//             [| pythonPath
//                Path.Combine(pythonPath, "python37.zip")
//                Path.Combine(pythonPath, "Lib\\site-packages")
//                Path.Combine(pythonPath, "python_code")
//                Path.Combine(pythonPath, "python_code\\DLKcat\\DeeplearningApproach\\Code\\example") |]
//         )
//     PythonEngine.Initialize()

// let runPythonCode (pythonPath: string) : unit =
//     initializePython pythonPath
//     use python = Py.CreateScope()

//     use _open =
//         new StreamReader(
//             Path.Combine(pythonPath, "python_code\\DLKcat\\DeeplearningApproach\\Code\\example\\test_2.py")
//         )

//     let _code = _open.ReadToEnd()

//     python.Exec _code |> ignore

//     let DLKcat =
//         (python.Get "DLKcatPrediction").Invoke [| pyObj "bin/Debug/net9.0/python_embedded/python_code" |] //DLKcatPrediction(pythonPath)

//     let result =
//         DLKcat.InvokeMethod(
//             "predict_for_one",
//             [| pyObj "L-Arginine"
//                pyObj "C(CC(C(=O)O)N)CN=C(N)N"
//                pyObj
//                    "MSLGIRYLALLPLFVITACQQPVNYNPPATQVAQVQPAIVNNSWIEISRSALDFNVKKVQSLLGKQSSLCAVLKGDAYGHDLSLVAPIMIENNVKCIGVTNNQELKEVRDLGFKGRLMRVRNATEQEMAQATNYNVEELIGDLDMAKRLDAIAKQQNKVIPIHLALNSGGMSRNGLEVDNKSGLEKAKQISQLANLKVVGIMSHYPEEDANKVREDLARFKQQSQQVLEVMGLERNNVTLHMANTFATITVPESWLDMVRVGGIFYGDTIASTDYKRVMTFKSNIASINYYPKGNTVGYDRTYTLKRDSVLANIPVGYADGYRRVFSNAGHALIAGQRVPVLGKTSMNTVIVDITSLNNIKPGDEVVFFGKQGNSEITAEEIEDISGALFTEMSILWGATNQRVLVD" |]
//         )

//     printfn "Result: %s" (result.As<string>())

//     DLKcat.InvokeMethod(
//         "predict_for_input",
//         [| pyObj "bin/Debug/net9.0/python_embedded/python_code/DLKcat/DeeplearningApproach/Code/example/input.tsv" |]
//     )
//     |> ignore

//     try PythonEngine.Shutdown() with _ -> ()

// [<EntryPoint>]
// let main argv =
//     let pythonPath = Path.Combine("C:\\Users\\Anyee\\source\\repos\\FsTest\\NetPython\\bin\\Debug\\net9.0\\python_embedded")
//     runPythonCode pythonPath
//     0
