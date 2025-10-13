package models

import (
	"context"
)

type Agent interface {
	Generate(context.Context, string) (any, error)
}
