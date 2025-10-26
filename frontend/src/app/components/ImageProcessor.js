import Image from "next/image";
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Skeleton } from './ui/skeleton';
import { Alert } from './ui/alert';
import { AlertCircle } from 'lucide-react';

// Helper function to capitalize first letter of each word (title case)
const toTitleCase = (str) => {
  const minorWords = ['a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'in', 'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet'];
  return str
    .split(' ')
    .map((word, index) => {
      const lowerWord = word.toLowerCase();
      // Always capitalize first and last word, or if not a minor word
      if (index === 0 || !minorWords.includes(lowerWord)) {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      }
      return lowerWord;
    })
    .join(' ');
};

// Helper function to clean up title
const cleanTitle = (title, year) => {
  if (!title) return '';

  // Replace dashes and underscores with spaces
  let cleaned = title.replace(/[-_]/g, ' ');

  // If year exists and matches last 4 characters of title, remove them
  if (year && cleaned.trim().slice(-4) === year) {
    cleaned = cleaned.trim().slice(0, -4).trim();
  }

  // Remove any trailing/leading whitespace and apply title case
  return toTitleCase(cleaned.trim());
};

// Helper function to capitalize artist name
const formatArtistName = (artist) => {
  if (!artist) return '';
  return artist
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};

// Helper function to format year
const formatYear = (year) => {
  if (!year) return null;
  const yearNum = parseInt(year, 10);
  if (isNaN(yearNum) || yearNum > 2025) {
    return 'Unknown';
  }
  return year;
};

export function ImageProcessor({ clipResults, cnnResults, loadingClip, loadingCnn, error }) {
  const hasResults = clipResults || cnnResults;
  const activeMethod = clipResults ? 'CLIP' : cnnResults ? 'CNN' : null;
  const results = clipResults || cnnResults || [];
  const isLoading = loadingClip || loadingCnn;

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2>Search Results</h2>
          <p className="text-muted-foreground">
            {hasResults
              ? `${results.length} similar artist${results.length !== 1 ? 's' : ''} found`
              : 'Click CLIP or CNN to search for similar artists'
            }
          </p>
        </div>
        <div className="text-muted-foreground">
          {activeMethod && (
            <p>Currently viewing: <span className="text-foreground">{activeMethod}</span></p>
          )}
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <div className="ml-2">
            <h4>Error</h4>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </Alert>
      )}

      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <Card key={i} className="p-4">
              <div className="flex gap-6">
                <Skeleton className="w-48 h-48 shrink-0" />
                <div className="flex-1 space-y-3">
                  <Skeleton className="h-6 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-4 w-1/3" />
                  <Skeleton className="h-4 w-2/3" />
                </div>
              </div>
            </Card>
          ))}
        </div>
      ) : hasResults ? (
        <div className="space-y-4">
          {activeMethod === 'CLIP' && results.map((result) => {
            const formattedYear = formatYear(result.year);
            const formattedTitle = cleanTitle(result.title, result.year);
            const formattedArtist = formatArtistName(result.artist);

            return (
              <Card key={result.rank} className="overflow-hidden hover:shadow-lg transition-shadow">
                <div className="flex flex-col md:flex-row gap-6 p-4">
                  {/* Image on the left */}
                  <div className="w-full md:w-48 h-48 shrink-0 overflow-hidden rounded-lg bg-muted">
                    <Image
                      src={result.cdn_url}
                      alt={formattedTitle}
                      width={192}
                      height={192}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* Details on the right */}
                  <div className="flex-1 space-y-3">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="secondary">Rank #{result.rank}</Badge>
                          <Badge variant="outline">Score: {result.score.toFixed(3)}</Badge>
                        </div>
                        <h3 className="mt-2">{formattedTitle}</h3>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Artist:</span>
                        <span>{formattedArtist}</span>
                      </div>

                      {formattedYear && (
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Year:</span>
                          <span>{formattedYear}</span>
                        </div>
                      )}

                      {result.style && (
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Style:</span>
                          <span>{result.style}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
          {activeMethod === 'CNN' && cnnResults.map((result) => {
            const formattedYear = formatYear(result.year);
            const formattedTitle = cleanTitle(result.title, result.year);
            const formattedArtist = formatArtistName(result.artist);

            return (
              <Card key={result.rank} className="overflow-hidden hover:shadow-lg transition-shadow">
                <div className="flex flex-col md:flex-row gap-6 p-4">
                  {/* Image on the left */}
                  <div className="w-full md:w-48 h-48 shrink-0 overflow-hidden rounded-lg bg-muted">
                    <Image
                      src={result.cdn_url}
                      alt={formattedTitle}
                      width={192}
                      height={192}
                      className="w-full h-full object-cover"
                    />
                  </div>

                  {/* Details on the right */}
                  <div className="flex-1 space-y-3">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="secondary">Rank #{result.rank}</Badge>
                          <Badge variant="outline">Score: {result.score.toFixed(3)}</Badge>
                        </div>
                        <h3 className="mt-2">{formattedTitle}</h3>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-muted-foreground">Artist:</span>
                        <span>{formattedArtist}</span>
                      </div>

                      {formattedYear && (
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Year:</span>
                          <span>{formattedYear}</span>
                        </div>
                      )}

                      {result.style && (
                        <div className="flex items-center gap-2">
                          <span className="text-muted-foreground">Style:</span>
                          <span>{result.style}</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      ) : (
        <div className="flex items-center justify-center h-64">
          <div className="text-center text-muted-foreground">
            <p>No results yet. Click CLIP or CNN to start searching.</p>
          </div>
        </div>
      )}
    </div>
  );
}
