#ifndef FACE_EMBEDDING_H
#define FACE_EMBEDDING_H

#include <string>
#include <vector>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <limits> 
#include "log_macros.h"

const size_t MAX_EMBEDDINGS_PER_PERSON = 5;  // Limit to 5 embeddings per person

// Struct to hold the embeddings for a single person
struct FaceEmbedding {
    std::string name;                              // Name of the person
    std::vector<std::vector<int8_t>> embeddings;   // Multiple int8 feature vectors

    // Add a new embedding for this person, but limit the number to MAX_EMBEDDINGS_PER_PERSON
    void AddEmbedding(const std::vector<int8_t>& embedding) {
        if (embeddings.size() < MAX_EMBEDDINGS_PER_PERSON) {
            embeddings.push_back(embedding);
        } else {
            // You could log a message or handle the case when the limit is reached
            printf("Maximum embeddings reached for %s\n", name.c_str());
        }
    }
};

// Struct to hold the embeddings for multiple persons
struct FaceEmbeddingCollection {
    std::vector<FaceEmbedding> embeddings;

    // Add a new embedding for a person (multiple embeddings allowed)
    void AddEmbedding(const std::string& personName, const std::vector<int8_t>& faceEmbedding) {
        for (auto& embedding : embeddings) {
            if (embedding.name == personName) {
                embedding.AddEmbedding(faceEmbedding);
                return;
            }
        }
        // If person is not found, create a new entry with the first embedding
        FaceEmbedding newEmbedding{personName, {faceEmbedding}};
        embeddings.push_back(newEmbedding);
    }

    // Retrieve a face embedding by name
    const FaceEmbedding* GetEmbeddingByName(const std::string& personName) const {
        for (const auto& embedding : embeddings) {
            if (embedding.name == personName) {
                return &embedding;
            }
        }
        return nullptr; // Return null if not found
    }

    // Calculate Euclidean Distance between two vectors
    double CalculateEuclideanDistance(const std::vector<int8_t>& v1, const std::vector<int8_t>& v2) const {
        if (v1.size() != v2.size()) return std::numeric_limits<double>::infinity(); // Return a large value if sizes don't match

        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            sum += std::pow(static_cast<int>(v1[i]) - static_cast<int>(v2[i]), 2);
        }
        return std::sqrt(sum);
    }

    // Find the most similar embedding in the collection and return the person's name
    std::string FindMostSimilarEmbedding(const std::vector<int8_t>& targetEmbedding) const {
        double minDistance = std::numeric_limits<double>::infinity();
        std::string mostSimilarPerson;

        for (const auto& embedding : embeddings) {
            for (const auto& storedEmbedding : embedding.embeddings) {
                double distance = CalculateEuclideanDistance(targetEmbedding, storedEmbedding);
                if (distance < minDistance) {
                    minDistance = distance;
                    mostSimilarPerson = embedding.name;
                }
            }
        }

        if (minDistance == std::numeric_limits<double>::infinity()) {
            return "No similar embedding found!";
        }
        return mostSimilarPerson;
    }

    // Function to print all embeddings in the collection
    void PrintEmbeddings() const {
        for (const auto& embedding : embeddings) {
            // Log the name of the person
            info("Name: %s\n", embedding.name.c_str());

            // Log each embedding
            for (size_t i = 0; i < embedding.embeddings.size(); ++i) {
                info("Embedding %zu: ", i + 1);
                
                // Log the int8 feature values of the embedding
                for (const auto& value : embedding.embeddings[i]) {
                    info("%d ", static_cast<int>(value));  // Cast to int to display as an integer
                }
                info("\n");  // End the line after printing the embedding
            }
            info("------------------------\n");  // Separator between different embeddings
        }
    }
    
};

#endif // FACE_EMBEDDING_H
