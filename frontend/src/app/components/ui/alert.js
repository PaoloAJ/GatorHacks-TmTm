function getAlertClasses(variant = "default", className = "") {
  // Base classes
  const baseClasses = "relative w-full rounded-lg border px-4 py-3 text-sm grid has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] grid-cols-[0_1fr] has-[>svg]:gap-x-3 gap-y-0.5 items-start [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current";

  // Variant classes
  const variantClasses = {
    default: "bg-card text-card-foreground",
    destructive: "text-destructive bg-card [&>svg]:text-current *:data-[slot=alert-description]:text-destructive/90",
  };

  return `${baseClasses} ${variantClasses[variant] || variantClasses.default} ${className}`.trim();
}

function Alert({
  className = "",
  variant = "default",
  ...props
}) {
  return (
    <div
      data-slot="alert"
      role="alert"
      className={getAlertClasses(variant, className)}
      {...props}
    />
  );
}

function AlertTitle({ className = "", ...props }) {
  return (
    <div
      data-slot="alert-title"
      className={`col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight ${className}`.trim()}
      {...props}
    />
  );
}

function AlertDescription({
  className = "",
  ...props
}) {
  return (
    <div
      data-slot="alert-description"
      className={`text-muted-foreground col-start-2 grid justify-items-start gap-1 text-sm [&_p]:leading-relaxed ${className}`.trim()}
      {...props}
    />
  );
}

export { Alert, AlertTitle, AlertDescription };
