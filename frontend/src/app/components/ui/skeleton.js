function Skeleton({ className = "", ...props }) {
  return (
    <div
      data-slot="skeleton"
      className={`bg-accent animate-pulse rounded-md ${className}`.trim()}
      {...props}
    />
  );
}

export { Skeleton };
