"use client";

import { useState, createContext, useContext } from "react";

const TabsContext = createContext();

function Tabs({ className = "", defaultValue, value, onValueChange, children, ...props }) {
  const [selectedTab, setSelectedTab] = useState(value || defaultValue);

  const handleTabChange = (newValue) => {
    if (onValueChange) {
      onValueChange(newValue);
    }
    if (value === undefined) {
      setSelectedTab(newValue);
    }
  };

  const activeTab = value !== undefined ? value : selectedTab;

  return (
    <TabsContext.Provider value={{ activeTab, onTabChange: handleTabChange }}>
      <div
        data-slot="tabs"
        className={`flex flex-col gap-2 ${className}`.trim()}
        {...props}
      >
        {children}
      </div>
    </TabsContext.Provider>
  );
}

function TabsList({ className = "", children, ...props }) {
  return (
    <div
      data-slot="tabs-list"
      className={`bg-muted text-muted-foreground inline-flex h-9 w-fit items-center justify-center rounded-xl p-[3px] flex ${className}`.trim()}
      {...props}
    >
      {children}
    </div>
  );
}

function TabsTrigger({ className = "", value, children, ...props }) {
  const { activeTab, onTabChange } = useContext(TabsContext);
  const isActive = activeTab === value;

  return (
    <button
      type="button"
      data-slot="tabs-trigger"
      data-state={isActive ? "active" : "inactive"}
      onClick={() => onTabChange(value)}
      className={`inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-xl border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-[color,box-shadow] focus-visible:ring-[3px] focus-visible:outline-1 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:outline-ring ${
        isActive
          ? "bg-card dark:text-foreground text-foreground dark:border-input dark:bg-input/30"
          : "text-foreground dark:text-muted-foreground"
      } ${className}`.trim()}
      {...props}
    >
      {children}
    </button>
  );
}

function TabsContent({ className = "", value, children, ...props }) {
  const { activeTab } = useContext(TabsContext);

  if (activeTab !== value) {
    return null;
  }

  return (
    <div
      data-slot="tabs-content"
      className={`flex-1 outline-none ${className}`.trim()}
      {...props}
    >
      {children}
    </div>
  );
}

export { Tabs, TabsList, TabsTrigger, TabsContent };
