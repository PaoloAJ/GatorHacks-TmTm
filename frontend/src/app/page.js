"use client";

import { useState } from "react";
import Image from "next/image";
import { Upload, BookOpen } from "lucide-react";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { ImageProcessor } from "./components/ImageProcessor";
import { Documentation } from "./components/Documentation";

export default function App() {
  const [showDocs, setShowDocs] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [clipResults, setClipResults] = useState(null);
  const [cnnResults, setCnnResults] = useState(null);
  const [loadingClip, setLoadingClip] = useState(false);
  const [loadingCnn, setLoadingCnn] = useState(false);
  const [error, setError] = useState(null);

  if (showDocs) {
    return <Documentation onBackToHome={() => setShowDocs(false)} />;
  }

  const handleImageUpload = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result);
        setFileName(file.name);
      };
      reader.readAsDataURL(file);
      // Reset results when new image is uploaded
      setClipResults(null);
      setCnnResults(null);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setFileName("");
    setClipResults(null);
    setCnnResults(null);
    setError(null);
  };

  const handleClipSearch = async () => {
    if (!selectedFile) return;

    setLoadingClip(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        "https://imagesimilarity.up.railway.app/search?alpha=0.7&top_k=5",
        {
          method: "POST",
          mode: "cors",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server responded with status ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();

      if (data.results && Array.isArray(data.results)) {
        setClipResults(data.results);
        setCnnResults(null); // Clear CNN results when showing CLIP
      } else {
        throw new Error("Invalid response format from server");
      }
    } catch (error) {
      console.error("Error fetching CLIP results:", error);
      setError(
        `Failed to search with CLIP: ${error.message}. The API may be experiencing issues or CORS restrictions.`
      );
    } finally {
      setLoadingClip(false);
    }
  };

  const handleCnnSearch = async () => {
    if (!selectedFile) return;

    setLoadingCnn(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(
        "https://imagesimilarity.up.railway.app/search?alpha=0.3&top_k=5",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `Server responded with status ${response.status}: ${
            errorData.error || response.statusText
          }`
        );
      }

      const data = await response.json();

      if (data.results && Array.isArray(data.results)) {
        setCnnResults(data.results);
        setClipResults(null); // Clear CLIP results when showing CNN
      } else {
        throw new Error("Invalid response format from server");
      }
    } catch (error) {
      console.error("Error fetching CNN results:", error);
      setError(`Failed to search with CNN: ${error.message}.`);
    } finally {
      setLoadingCnn(false);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-purple-50 to-blue-50 overflow-hidden">
      <div className="flex flex-col lg:flex-row h-full">
        {/* Sidebar - Upload Form */}
        <div className="w-full lg:w-80 xl:w-96 bg-white border-r border-border flex flex-col h-full overflow-hidden">
          <div className="p-6 lg:p-8 flex flex-col gap-4 h-full justify-between">
            <div className="flex-shrink-0">
              <h1 className="text-3xl mb-2">𝔸ℝ𝕋-ificial</h1>
              <p className="text-sm text-muted-foreground mb-3">
                Upload an image to rediscover the true artist behind an AI
                generated image
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDocs(true)}
                className="gap-2"
              >
                <BookOpen className="w-4 h-4" />
                Docs
              </Button>
            </div>

            <Card className="p-4 flex-shrink-0">
              <div className="flex flex-col items-center justify-center gap-3">
                {selectedImage ? (
                  <>
                    <div className="w-full">
                      <h3 className="text-sm font-medium mb-1">
                        Original Image
                      </h3>
                      <p className="text-xs text-muted-foreground truncate mb-2">
                        {fileName}
                      </p>
                      <div className="relative aspect-square overflow-hidden rounded-lg bg-muted max-h-48">
                        <Image
                          src={selectedImage}
                          alt="Original"
                          fill
                          className="object-cover"
                        />
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                      <Upload className="w-8 h-8 text-primary" />
                    </div>
                    <div className="text-center">
                      <h3 className="text-sm font-medium mb-1">
                        Upload your image
                      </h3>
                      <p className="text-xs text-muted-foreground mb-2">
                        JPG, PNG, or GIF formats
                      </p>
                    </div>
                  </>
                )}
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
                <Button
                  className="cursor-pointer w-full"
                  onClick={() =>
                    document.getElementById("image-upload").click()
                  }
                >
                  {selectedImage ? "Upload New Image" : "Choose Image"}
                </Button>
                {selectedImage && (
                  <Button
                    variant="outline"
                    onClick={handleReset}
                    className="w-full"
                  >
                    Reset
                  </Button>
                )}
              </div>
            </Card>

            {selectedImage && (
              <div className="flex-shrink-0">
                <h3 className="text-sm font-medium mb-2">Search Methods</h3>
                <div className="flex gap-2">
                  <Button
                    onClick={handleClipSearch}
                    disabled={loadingClip || loadingCnn}
                    className="flex-1"
                  >
                    {loadingClip ? "Searching..." : "CLIP"}
                  </Button>
                  <Button
                    onClick={handleCnnSearch}
                    disabled={loadingClip || loadingCnn}
                    className="flex-1"
                  >
                    {loadingCnn ? "Searching..." : "CNN"}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Search may take up to 10 seconds
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Main Content - Results */}
        <div className="flex-1 h-full overflow-y-auto">
          <div className="p-6 lg:p-8">
            {selectedImage ? (
              <ImageProcessor
                clipResults={clipResults}
                cnnResults={cnnResults}
                loadingClip={loadingClip}
                loadingCnn={loadingCnn}
                error={error}
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-muted-foreground max-w-md">
                  <Upload className="w-16 h-16 mx-auto mb-4 opacity-20" />
                  <h3 className="mb-2">No image uploaded yet</h3>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
