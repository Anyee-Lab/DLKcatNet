module DLKcatService

open System
open System.IO
open Python.Runtime

type predict_type =
    { name: string
      smlies: string
      sequence: string }

let pyObj (value: obj) : PyObject =
    match value with
    | :? string as s -> new PyString(s) :> PyObject
    | :? int as i -> new PyInt(i) :> PyObject
    | _ -> failwith "Unsupported type"

type DLKcatPrediction(pythonPath: string, dlkcatPath: string) =
    let DLKcat =
        Runtime.PythonDLL <- Path.Combine(pythonPath, "python37.dll")

        PythonEngine.PythonPath <-
            String.Join(
                string Path.PathSeparator,
                [| pythonPath
                   Path.Combine(pythonPath, "python37.zip")
                   Path.Combine(pythonPath, "Lib\\site-packages")
                   Path.Combine(dlkcatPath, "DeeplearningApproach\\Code\\example") |]
            )

        PythonEngine.Initialize()
        printfn "[DotNet]: Python Engine is Initialized!"

        let python = Py.CreateScope()

        let _open =
            new StreamReader(Path.Combine(dlkcatPath, "DeeplearningApproach\\Code\\example\\dlkcat_net.py"))

        let _code = _open.ReadToEnd()

        python.Exec _code |> ignore

        let a = (python.Get "DLKcatPrediction").Invoke [| pyObj dlkcatPath |] //DLKcatPrediction(pythonPath)
        printfn "[DotNet]: DLKcat is Initialized!"
        a


    member this.predictForOne (name: string) (smlies: string) (sequence: string) =
        let result =
            DLKcat.InvokeMethod("predict_for_one", [| pyObj name; pyObj smlies; pyObj sequence |])

        result.As<string>()

    member this.predictForInput(data: predict_type) =
        let result = this.predictForOne data.name data.smlies data.sequence
        result

    member this.shutdown() =
        try
            PythonEngine.Shutdown()
        with _ ->
            ()

        printfn "[DotNet]: Python Engine is Shutdown!"
