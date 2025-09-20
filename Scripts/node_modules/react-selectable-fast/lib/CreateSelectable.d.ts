import React from 'react';
import { TSelectableItemState, TSelectableItemProps } from './Selectable.types';
export declare const createSelectable: <T extends any>(WrappedComponent: React.ComponentType<TSelectableItemState & {
    selectableRef(node: HTMLElement | null): void;
} & T>) => React.ComponentType<T & Partial<Pick<TSelectableItemProps, "isSelected">>>;
