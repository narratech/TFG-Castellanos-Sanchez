
#include "OneHotEncoder.h"

TArray<float> UOneHotEncoder::OneHotEncodeWithCategories(
	const FString& Input,
	const TArray<FString>& Categories
)
{

	// Copiar categories para no modificar el original
	TArray<FString> SortedCategories = Categories;

	// Ordenar alfabéticamente
	SortedCategories.Sort([](const FString& A, const FString& B)
		{
			return A < B; // Orden ascendente
		});

	const int32 NumCategories = SortedCategories.Num();

	TArray<float> OneHotVector;
	OneHotVector.Init(0, NumCategories);

	const int32 Index = SortedCategories.IndexOfByKey(Input);

	if (Index != INDEX_NONE)
	{
		OneHotVector[Index] = 1;
	}
	// Si no se encuentra, queda todo en 0 (categoría desconocida)

	return OneHotVector;
}

FString UOneHotEncoder::OneHotDecodeWithCategories(
	const TArray<float>& Input,
	const TArray<FString>& Categories
)
{

	// Copiar categories para no modificar el original
	TArray<FString> SortedCategories = Categories;

	// Ordenar alfabéticamente
	SortedCategories.Sort([](const FString& A, const FString& B)
		{
			return A < B; // Orden ascendente
		});

	const int32 NumCategories = SortedCategories.Num();

	FString OneHot;

	for (int i = 0; i < Input.Num(); i++) {
		if (Input[i] == 1) {
			return Categories[i];
		}
	}

	//Vacio en caso de todo a 0
	return "";
}

