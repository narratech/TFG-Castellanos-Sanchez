// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.IO;

public class TFGProject : ModuleRules
{
	public TFGProject(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });

		PrivateDependencyModuleNames.AddRange(new string[] {  });

        // Ruta de ONNX
        string ONNXPath = Path.Combine(ModuleDirectory, "../../ThirdParty/ONNXRuntime");

        //Includes
        PublicIncludePaths.Add(Path.Combine(ONNXPath, "include"));

        //Static libraries
        PublicAdditionalLibraries.Add(Path.Combine(ONNXPath, "lib", "onnxruntime.lib"));


        //Dynamic libraries
        PublicDelayLoadDLLs.Add("onnxruntime.dll");
        RuntimeDependencies.Add(Path.Combine(ONNXPath, "lib", "onnxruntime.dll"));

        // Uncomment if you are using Slate UI
        // PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
    }
}
