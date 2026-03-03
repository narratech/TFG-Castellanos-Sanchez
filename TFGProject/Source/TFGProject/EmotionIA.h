#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"

#include "EmotionIA.generated.h"

struct FEmotionIAInternalModel;

UCLASS(ClassGroup = (AI), meta = (BlueprintSpawnableComponent))
class MYPROJECT_API UEmotionIA : public UActorComponent
{
    GENERATED_BODY()

public:

    UFUNCTION(BlueprintCallable, Category = "EmotionAI")
    bool InitModel();

    UFUNCTION(BlueprintCallable, Category = "EmotionAI")
    TArray<float> RunInference(const TArray<float>& InputData);

protected:

    virtual void BeginDestroy() override;
    virtual void BeginPlay() override;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EmotionAI")
    FString ModelPath;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EmotionAI")
    int64 BatchSize = 1;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EmotionAI")
    int64 SequenceLength = 35;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EmotionAI")
    int32 OrdinalsSize = 0;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "EmotionAI")
    TArray<int64> OnehotSizes;
private:

    FEmotionIAInternalModel* InternalModel = nullptr;

    int64 FeatureSize;

    bool CheckONNXDependenciesDynamic();

    TArray<float> InputSequence;

    int64 CircularIndex;

    TArray<float> LinearInput;

    TArray<float> Output;
};