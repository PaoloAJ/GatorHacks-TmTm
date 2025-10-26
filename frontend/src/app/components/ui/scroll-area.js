"use client";

function ScrollArea({ className = "", children, ...props }) {
  return (
    <div
      data-slot="scroll-area"
      className={`relative overflow-auto ${className}`.trim()}
      {...props}
    >
      {children}
    </div>
  );
}

function ScrollBar({ className = "", orientation = "vertical", ...props }) {
  return null; // Native scrollbar will be used
}

export { ScrollArea, ScrollBar };
