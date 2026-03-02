
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

