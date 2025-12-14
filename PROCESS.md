

**Main
	APPLY
	MODEL
		- LLM 
		- MISTRAL
		- LLaVA  processLLaVA
	EVAL


**processLLaVA:
	checkStage->stage
	stage =
	0->processStage0
	1->processStage1
	2->processStage2

**processStage0:
buildContentProcess()  devuelve la lista de imagenes y el documento texto asociado
extractFeatures()
	mode(clip, dino, VIT)
	mode VIT:
		model, preprocess = load model
		forimage:
			preprocess
			torch, model.encode->features
			
	for image
		build context, keywords, clusterLabel
	
	processPrompt1()
	results = EVALMODEL
	for image: asociar result a image
	write json
	
processPrompt1()
"prompt": "<Para esta fotografía que enfoca un detalle> representando a <'Recinto funerario de Diofantes'>, Ten en cuenta sin mencionar explícitamente que <es un yacimiento arqueológico con su elemento principal no son sólo piedras> y describe en castellano solo lo visible. Máximo 20 palabras, cíñete a estos dos items:\nItem 1 <**Ubicación y entorno**: Describe el entorno y cómo se ubica el elemento principal.> Item 2 <**Elemento principal que protagoniza la imagen**: Explica la estuctura de el elemento principal.> Máximo 20 palabras, no mencionar deteriorado y/o antiguo."
	
**processStage1:
	processPrompt2()
	EVALMODEL
	for image: asociar result a image
	write json
	
processPrompt2()
"Este es el contexto con el que vamos a enriquecer una descripción de imagen CONTEXTO: '', mejora la redacción componiendo un texto contínuo y con sentido global Limita las respuestas a 40 palabras como máximo, pero si alcanzas ese límite a mitad de frase, puedes extenderte hasta completarla (hasta el punto final). Utiliza solo la información del CONTEXTO sin añadir nada más. "

**
			
