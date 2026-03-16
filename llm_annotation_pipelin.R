#---1. Load required libraries---#

library(pdftools)
library(dplyr)
library(stringr)
library(stringi)
library(rollama)

#---2. Text extraction---#

#Define function for text extraction from PDF files

text_extract <- function(filename){
	text <- pdf_text(filename)
	text <- paste(text, collapse = " ")
	text <- iconv(text, from="latin1", to="UTF-8")

	#chose markers according to the structure of the text
	parts <- strsplit(text, "MARKER1")
	text <- parts [[1]][2]
	parts <- strsplit(text, "MARKER2")

	text <- parts [[1]][1]
	df <- data.frame(
		#use manipulated filenames as id
		id=gsub("_NP.pdf","",as.character(filename)),
		text=text
		)
}


#List all PDF files (choose correct directory)

files <- list.files(pattern = "pdf$")

#Apply function to all PDF files

texts <-  lapply(files, text_extract)

#Create final data

texts <- do.call(rbind, texts)

#---3. Data Preparation and Prompts---#

#Load and format data

reference <- #ANNOTATION FILE#
reference <- reference[order(reference$id),]
texts <- texts[order(texts$id),]

anamnesis <- tibble(
	id = texts$id,
	text = texts$text
)

#Chose item from reference
ref <- reference$nausea 
ref <- tolower(ref)

#Define system and user prompt, code UTF-8

systemmsg <- "Du hilfst als KI-Assistent aus medizinischen Texten Informationen zu extrahieren."

systemmsg <- iconv(systemmsg, from="latin1", to="UTF-8")

classification_question <- "Leidet der Patient an Übelkeit? Dann antworte mit #ja#. Wenn keine Übelkeit vorliegt oder keine Information dazu vorhanden ist, antworte mit #nein#. Erbrechen deutet auf Übelkeit hin. Gib keine Begründung aus."

classification_question <- iconv(classification_question, from="latin1", to="UTF-8")

#Prepare queries for the LLM prompting

queries <- make_query(
    text = anamnesis$text,
    prompt = classification_question,
    template = "text: {text} \n {prompt}",
    system = systemmsg
)

#---4. Prompting---#

#Define function for prompting

annotate <- function() {

	#Record start time
	start_time <- Sys.time()

	#Send queries to the defined model
	llm <- query(
		queries,
		model = "gemma2:27b",
		model_params = list(temperature = 0.0),
		screen = FALSE, output = "text"
	)

	#Record end time and calculate time needed
	end_time <- Sys.time()
	time <- end_time - start_time
	time <- as.numeric(time, units = "secs")

	#Clean LLM output and force output to yes/no
	llm <- tolower(llm)
	llm <- gsub("[\r\n]", "", llm)
	llm <- gsub("#", "", as.character(llm))
	llm <- gsub(" ", "", llm, fixed = TRUE)
	llm[!llm %in% c("ja", "nein")] <- "nein"

	#Calculate performance metrics
	acc <- sum(llm == ref)
	TP <- sum(llm == "ja" & ref == "ja")
	FP <- sum(llm == "ja" & ref == "nein")
	TN <- sum(llm == "nein" & ref == "nein")
	FN <- sum(llm == "nein" & ref == "ja")
	sens <- TP / (TP + FN)
	spec <- TN / (TN + FP)
	ppv <- TP / (TP + FP)
	npv <- TN / (TN + FN)
	f1 <- 2 * (ppv * sens) / (ppv + sens)

	#Return results as list
	return(list(
	llm = llm,
	time = time,
	acc = acc,	
	sens = sens,
	spec = spec,
	ppv = ppv,
	npv = npv,
	f1 = f1 ))
}

#Run iterations

iterations <- 10
results <- list()

for (i in 1:iterations) {
    result <- annotate()
    results[[i]] <- result
}

#---5. Create final output and save as CSV file---#

#create final dataframe

df <- data.frame(id = c(anamnesis$id, "time", "acc", "sens", "spec", "ppv", "npv", "f1"))

for (i in 1:iterations) {
         colname <- paste0("llm_", i)
         df[[colname]] <- c(
		results[[i]]$llm,
		results[[i]]$time,
		results[[i]]$acc,
		results[[i]]$sens,
		results[[i]]$spec,
		results[[i]]$ppv,
		results[[i]]$npv,
		results[[i]]$f1)
     }

#save as CSV file

write.csv(df, "FILENAME.csv")

