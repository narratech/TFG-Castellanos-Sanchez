#include "EmotionIA.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"
#include "onnxruntime_c_api.h"
#include "HAL/PlatformProcess.h"
#include "Windows/WindowsHWrapper.h" // para LoadLibraryEx
#include "Windows/AllowWindowsPlatformTypes.h"
#include <vector>

#define ORT_VERSION 20

// -------------------------
// Internal model (C API)
// -------------------------

struct FEmotionIAInternalModel
{
    TArray<uint8> ModelBuffer;
    OrtEnv* Env = nullptr;
    OrtSessionOptions* SessionOptions = nullptr;
    OrtSession* Session = nullptr;

    ~FEmotionIAInternalModel()
    {
        const OrtApi* Api = OrtGetApiBase()->GetApi(ORT_VERSION);
        if (Session) { Api->ReleaseSession(Session); Session = nullptr; }
        if (SessionOptions) { Api->ReleaseSessionOptions(SessionOptions); SessionOptions = nullptr; }
        if (Env) { Api->ReleaseEnv(Env); Env = nullptr; }
    }
};

bool UEmotionIA::CheckONNXDependenciesDynamic()
{
    FString BasePath = FPaths::ProjectDir() / TEXT("ThirdParty/ONNXRuntime/lib");
    FString MainDLL = BasePath / TEXT("onnxruntime.dll");

    // Intentamos cargar la DLL principal sin ejecutar DllMain
    HMODULE MainHandle = LoadLibraryExW(*MainDLL, nullptr, LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    if (!MainHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("No se pudo cargar la DLL principal: %s"), *MainDLL);
        return false;
    }

    // Si llegamos aquí, podemos usar Dependency Walker / DumpBin para inspeccionar dependencias reales.
    // Alternativa manual: verificar DLL comunes (MSVC runtimes) como antes:
    TArray<FString> CommonDependencies = {
        TEXT("KERNEL32.dll"),
        TEXT("ADVAPI32.dll"),
        TEXT("MSVCP140.dll"),
        TEXT("MSVCP140_1.dll"),
        TEXT("api-ms-win-core-path-l1-1-0.dll"),
        TEXT("dbghelp.dll"),
        TEXT("SETUPAPI.dll"),
        TEXT("dxgi.dll"),
        TEXT("VCRUNTIME140_1.dll"),
        TEXT("VCRUNTIME140.dll"),
        TEXT("api-ms-win-crt-heap-l1-1-0.dll"),
        TEXT("api-ms-win-crt-runtime-l1-1-0.dll"),
        TEXT("api-ms-win-crt-convert-l1-1-0.dll"),
        TEXT("api-ms-win-crt-stdio-l1-1-0.dll"),
        TEXT("api-ms-win-crt-string-l1-1-0.dll"),
        TEXT("api-ms-win-crt-time-l1-1-0.dll"),
        TEXT("api-ms-win-crt-filesystem-l1-1-0.dll"),
        TEXT("api-ms-win-crt-locale-l1-1-0.dll"),
        TEXT("api-ms-win-crt-math-l1-1-0.dll")
    };

    bool bAllLoaded = true;

    for (const FString& Dep : CommonDependencies)
    {
        HMODULE Handle = LoadLibraryW(*Dep);
        if (!Handle)
        {
            UE_LOG(LogTemp, Error, TEXT("Falta DLL dependiente: %s"), *Dep);
            bAllLoaded = false;
        }
    }

    // Liberamos la DLL principal
    FreeLibrary(MainHandle);

    if (!bAllLoaded)
    {
        UE_LOG(LogTemp, Error, TEXT("No todas las dependencias están presentes."));
        return false;
    }

    return true;
}

// -------------------------
// UEmotionIA Implementation
// -------------------------

bool UEmotionIA::InitModel()
{
    // Lista de DLLs que normalmente necesita ONNX Runtime C API
    if (!CheckONNXDependenciesDynamic()) {
        return false;
    }

    // Paso 1: Crear InternalModel
    InternalModel = new FEmotionIAInternalModel();
    if (!InternalModel)
    {
        UE_LOG(LogTemp, Error, TEXT("No se pudo crear InternalModel"));
        return false;
    }

    // Paso 2: Cargar modelo desde disco
    TArray<uint8> Buffer;
    FString FullPath = FPaths::Combine(FPaths::ProjectContentDir(), ModelPath);

    if (!FFileHelper::LoadFileToArray(Buffer, *FullPath) || Buffer.Num() == 0)
    {
        UE_LOG(LogTemp, Error, TEXT("Modelo no encontrado o vacío en %s"), *FullPath);
        delete InternalModel;
        InternalModel = nullptr;
        return false;
    }

    InternalModel->ModelBuffer = Buffer;

    // Paso 3: Obtener API de ONNX Runtime
    const OrtApiBase* Base = OrtGetApiBase();

    if (!Base)
    {
        UE_LOG(LogTemp, Error, TEXT("OrtGetApiBase devolvió NULL"));
        delete InternalModel;
        InternalModel = nullptr;
        return false;
    }

    const OrtApi* Api = Base->GetApi(ORT_VERSION);

    if (!Api)
    {
        UE_LOG(LogTemp, Error, TEXT("GetApi devolvió NULL"));
        delete InternalModel;
        InternalModel = nullptr;
        return false;
    }

    // Paso 4: Crear Env
    OrtStatus* Status = Api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "EmotionAI", &InternalModel->Env);
    if (Status)
    {
        UE_LOG(LogTemp, Error, TEXT("Error creando OrtEnv"));
        delete InternalModel;
        InternalModel = nullptr;
        return false;
    }

    // Paso 5: Crear SessionOptions
    Status = Api->CreateSessionOptions(&InternalModel->SessionOptions);
    if (Status)
    {
        UE_LOG(LogTemp, Error, TEXT("Error creando SessionOptions"));
        delete InternalModel;
        InternalModel = nullptr;
        return false;
    }

    Api->SetIntraOpNumThreads(InternalModel->SessionOptions, 1);
    Api->SetSessionGraphOptimizationLevel(InternalModel->SessionOptions, ORT_ENABLE_EXTENDED);

    const wchar_t* ModelPathW = *FullPath;
    // Paso 6: Crear Session desde el buffer en memoria
    Status = Api->CreateSession(
        InternalModel->Env,
        ModelPathW,
        InternalModel->SessionOptions,
        &InternalModel->Session
    );

    if (Status != nullptr)
    {
        const char* ErrorMsg = Api->GetErrorMessage(Status);

        UE_LOG(
            LogTemp,
            Error,
            TEXT("Error creando OrtSession desde memoria: %s"),
            ANSI_TO_TCHAR(ErrorMsg)
        );

        Api->ReleaseStatus(Status);

        delete InternalModel;
        InternalModel = nullptr;
        return false;
    }

    return true;
}

TArray<float> UEmotionIA::RunInference(const TArray<float>& InputData)
{
    if (!InternalModel || !InternalModel->Session)
    {
        UE_LOG(LogTemp, Error, TEXT("Modelo no inicializado"));
        return Output;
    }

    if (InputData.Num() != FeatureSize)
    {
        UE_LOG(LogTemp, Error, TEXT("Input incorrecto: el input fue de tamaño %d"), InputData.Num());
        return Output;
    }

    // Comprobar que la entrada no es la misma que la anterior (Quitar y delegar al desarrollador)
    bool InputIsLastOne = true;
    int64 prevIndex = CircularIndex - 1;
    if (prevIndex == -1)
        prevIndex = SequenceLength - 1;

    for (int i = prevIndex * FeatureSize; i < (prevIndex + 1) * FeatureSize; i++) {
        if (InputSequence[i] != InputData[i % FeatureSize]) {
            InputIsLastOne = false;
            break;
        }
    }

    // Devolvemos el mismo output que en la inferencia anterior
    if (InputIsLastOne)
        return Output;

    // Sobreescribir la ultima entrada registrada
    for (int i = CircularIndex * FeatureSize; i < (CircularIndex + 1) * FeatureSize; i++) {
        InputSequence[i] = InputData[i % FeatureSize];
    }

    //Obtener la API de Ort
    const OrtApi* Api = OrtGetApiBase()->GetApi(ORT_VERSION);

    // Crear MemoryInfo para CPU
    OrtMemoryInfo* MemoryInfo = nullptr;
    Api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &MemoryInfo);

    // Forma del tensor
    TArray<int64> InputShape = { BatchSize, SequenceLength, FeatureSize };

    for (int32 i = 0; i < SequenceLength; ++i)
    {
        int32 SrcIndex = ((CircularIndex - i + SequenceLength) % SequenceLength) * FeatureSize;
        FMemory::Memcpy(
            LinearInput.GetData() + (SequenceLength - 1 - i) * FeatureSize,
            InputSequence.GetData() + SrcIndex,
            FeatureSize * sizeof(float)
        );
    }

    // Crear tensor de entrada con los datos de InputData
    OrtValue* InputTensor = nullptr;
    Api->CreateTensorWithDataAsOrtValue(
        MemoryInfo,
        const_cast<float*>(InputSequence.GetData()),   // Datos
        InputSequence.Num() * sizeof(float),          // Tamaño en bytes
        InputShape.GetData(),                      // Forma
        InputShape.Num(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,       // Tipo
        &InputTensor
    );

    // Nombres de input/output
    OrtAllocator* Alloc = nullptr;
    Api->GetAllocatorWithDefaultOptions(&Alloc);

    char* InputName = nullptr;
    Api->SessionGetInputName(InternalModel->Session, 0, Alloc, &InputName);

    char* OutputName = nullptr;
    Api->SessionGetOutputName(InternalModel->Session, 0, Alloc, &OutputName);

    // Run
    const char* InputNames[] = { InputName };
    const char* OutputNames[] = { OutputName };
    OrtValue* OutputTensors[1] = { nullptr };

    Api->Run(
        InternalModel->Session,
        nullptr,             // RunOptions
        InputNames,
        &InputTensor,
        1,                   // num_inputs
        OutputNames,
        1,                   // num_outputs
        OutputTensors
    );

    // Obtener datos de salida
    float* FloatArray = nullptr;
    Api->GetTensorMutableData(OutputTensors[0], (void**)&FloatArray);

    for (int i = 0; i < 6; i++)
    {
        Output[i] = FloatArray[i];
    }


    // Liberar memoria temporal
    Api->ReleaseValue(InputTensor);
    Api->ReleaseValue(OutputTensors[0]);
    Api->AllocatorFree(Alloc, InputName);
    Api->AllocatorFree(Alloc, OutputName);
    Api->ReleaseMemoryInfo(MemoryInfo);

    CircularIndex++;
    if (CircularIndex == SequenceLength)
        CircularIndex = 0;

    return Output;
}

void UEmotionIA::BeginDestroy()
{
    if (InternalModel)
    {
        delete InternalModel;
        InternalModel = nullptr;
    }

    Super::BeginDestroy();
}

void UEmotionIA::BeginPlay()
{
    Super::BeginPlay();

    if (ModelPath.IsEmpty())
    {
        UE_LOG(LogTemp, Error, TEXT("EmotionIA: ModelPath no está configurado"));
        return;
    }

    if (!InitModel())
    {
        UE_LOG(LogTemp, Error, TEXT("EmotionIA: No se pudo inicializar el modelo"));
        return;
    }

    CircularIndex = 0;

    InputSequence.Init(0, FeatureSize * SequenceLength);
    LinearInput.Init(0, FeatureSize * SequenceLength);
    Output.Init(0,6);
}